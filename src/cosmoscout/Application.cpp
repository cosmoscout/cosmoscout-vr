////////////////////////////////////////////////////////////////////////////////////////////////////
//                               This file is part of CosmoScout VR                               //
//      and may be used under the terms of the MIT license. See the LICENSE file for details.     //
//                        Copyright: (c) 2019 German Aerospace Center (DLR)                       //
////////////////////////////////////////////////////////////////////////////////////////////////////

#include "Application.hpp"

#include "../cs-core/DragNavigation.hpp"
#include "../cs-core/GraphicsEngine.hpp"
#include "../cs-core/GuiManager.hpp"
#include "../cs-core/InputManager.hpp"
#include "../cs-core/PluginBase.hpp"
#include "../cs-core/Settings.hpp"
#include "../cs-core/SolarSystem.hpp"
#include "../cs-core/TimeControl.hpp"
#include "../cs-graphics/MouseRay.hpp"
#include "../cs-utils/Downloader.hpp"
#include "../cs-utils/convert.hpp"
#include "../cs-utils/filesystem.hpp"
#include "../cs-utils/logger.hpp"
#include "../cs-utils/utils.hpp"
#include "GetSelectionStateNode.hpp"
#include "ObserverNavigationNode.hpp"

#include <VistaBase/VistaTimeUtils.h>
#include <VistaInterProcComm/Cluster/VistaClusterDataSync.h>
#include <VistaKernel/Cluster/VistaClusterMode.h>
#include <VistaKernel/DisplayManager/GlutWindowImp/VistaGlutWindowingToolkit.h>
#include <VistaKernel/DisplayManager/VistaDisplayManager.h>
#include <VistaKernel/EventManager/VistaEventManager.h>
#include <VistaKernel/EventManager/VistaSystemEvent.h>
#include <VistaKernel/GraphicsManager/VistaOpenGLNode.h>
#include <VistaKernel/GraphicsManager/VistaSceneGraph.h>
#include <VistaKernel/GraphicsManager/VistaTransformNode.h>
#include <VistaKernel/VistaSystem.h>
#include <VistaKernelOpenSGExt/VistaOpenSGMaterialTools.h>
#include <VistaOGLExt/VistaShaderRegistry.h>
#include <curlpp/cURLpp.hpp>
#include <spdlog/spdlog.h>

////////////////////////////////////////////////////////////////////////////////////////////////////

#ifdef __linux__
#define OPENLIB(libname) dlopen((libname), RTLD_LAZY)
#define LIBFUNC(handle, fn) dlsym((handle), (fn))
#define CLOSELIB(handle) dlclose((handle))
#define LIBERROR() dlerror()
#define LIBFILETYPE ".so"
#define PLUGIN_PATH "../share/plugins/"
#else
#define OPENLIB(libname) LoadLibrary((libname))
#define LIBFUNC(handle, fn) GetProcAddress((HMODULE)(handle), (fn))
#define CLOSELIB(handle) FreeLibrary((HMODULE)(handle))
#define LIBERROR() GetLastError()
#define LIBFILETYPE ".dll"
#define PLUGIN_PATH "..\\share\\plugins\\"
#endif

////////////////////////////////////////////////////////////////////////////////////////////////////

Application::Application(cs::core::Settings const& settings)
    : VistaFrameLoop()
    , mSettings(std::make_shared<cs::core::Settings>(settings)) {

  // Initialize curl.
  cURLpp::initialize();
}

////////////////////////////////////////////////////////////////////////////////////////////////////

Application::~Application() {
  // Last but not least, cleanup curl.
  cURLpp::terminate();
}

////////////////////////////////////////////////////////////////////////////////////////////////////

bool Application::Init(VistaSystem* pVistaSystem) {

  // Make sure that our shaders are found by ViSTA.
  VistaShaderRegistry::GetInstance().AddSearchDirectory("../share/resources/shaders");

  // First we create all our core classes.
  mInputManager   = std::make_shared<cs::core::InputManager>();
  mFrameTimings   = std::make_shared<cs::utils::FrameTimings>();
  mGraphicsEngine = std::make_shared<cs::core::GraphicsEngine>(mSettings);
  mGuiManager     = std::make_shared<cs::core::GuiManager>(mSettings, mInputManager, mFrameTimings);
  mSceneSync =
      std::unique_ptr<IVistaClusterDataSync>(GetVistaSystem()->GetClusterMode()->CreateDataSync());
  mTimeControl = std::make_shared<cs::core::TimeControl>(mSettings);
  mSolarSystem = std::make_shared<cs::core::SolarSystem>(
      mSettings, mFrameTimings, mGraphicsEngine, mTimeControl);
  mDragNavigation.reset(new cs::core::DragNavigation(mSolarSystem, mInputManager, mTimeControl));

  // The ObserverNavigationNode is used by several DFN networks to move the celestial observer.
  VdfnNodeFactory* pNodeFactory = VdfnNodeFactory::GetSingleton();
  pNodeFactory->SetNodeCreator(
      "ObserverNavigationNode", new ObserverNavigationNodeCreate(mSolarSystem.get()));
  pNodeFactory->SetNodeCreator(
      "GetSelectionStateNode", new GetSelectionStateNodeCreate(mInputManager.get()));

  // This connects several parts of CosmoScout VR to each other.
  connectSlots();

  // Setup user interface callbacks.
  registerGuiCallbacks();

  // initialize the mouse pointer state ------------------------------------------------------------

  // If we are running on freeglut, we can hide the mouse pointer when the mouse ray should be
  // shown. This is determined by the settings key "enableMouseRay".
  auto windowingToolkit = dynamic_cast<VistaGlutWindowingToolkit*>(
      GetVistaSystem()->GetDisplayManager()->GetWindowingToolkit());

  if (windowingToolkit) {
    for (auto const& window : GetVistaSystem()->GetDisplayManager()->GetWindows()) {
      windowingToolkit->SetCursorIsEnabled(window.second, !mSettings->mEnableMouseRay);
    }
  }

  // If the settings key "enableMouseRay" is set to true, we add a cone geometry to the
  // SELECTION_NODE. The SELECTION_NODE is controlled by the users input device (via the DFN).
  if (mSettings->mEnableMouseRay) {
    auto                sceneGraph = GetVistaSystem()->GetGraphicsManager()->GetSceneGraph();
    VistaTransformNode* pIntentionNode =
        dynamic_cast<VistaTransformNode*>(sceneGraph->GetNode("SELECTION_NODE"));

    VistaTransformNode* mRayTrans = sceneGraph->NewTransformNode(pIntentionNode);
    mRayTrans->SetScale(0.001f, 0.001f, 30.0f);
    mRayTrans->SetName("Ray_Trans");

    auto ray      = new cs::graphics::MouseRay();
    auto coneNode = sceneGraph->NewOpenGLNode(mRayTrans, ray);
    coneNode->SetName("Cone_Node");

    VistaOpenSGMaterialTools::SetSortKeyOnSubtree(
        pIntentionNode, static_cast<int>(cs::utils::DrawOrder::eRay));
  } else {
    cs::core::GuiManager::setCursor(cs::gui::Cursor::ePointer);
  }

  // Initialize some gui components
  if (!mSettings->mEnableSensorSizeControl) {
    mGuiManager->getGui()->executeJavascript(
        "document.querySelector('#enableSensorSizeControl').classList.add('hidden')");
  }

  if (!mSettings->mEnableHDR.value_or(false)) {
    mGuiManager->getGui()->callJavascript(
        "CosmoScout.gui.setCheckboxValue", "graphics.setEnableHDR", false, true);
  }

  mGuiManager->enableLoadingScreen(true);

  // open plugins ----------------------------------------------------------------------------------
  for (auto const& plugin : mSettings->mPlugins) {
    openPlugin(plugin.first);
  }

  return VistaFrameLoop::Init(pVistaSystem);
}

////////////////////////////////////////////////////////////////////////////////////////////////////

void Application::Quit() {

  // Do not attempt to print anything to the on-screen console.
  cs::utils::logger::onMessage().disconnect(mOnMessageConnection);

  // De-init all plugins first.
  for (auto const& plugin : mPlugins) {
    plugin.second.mPlugin->deInit();
  }

  // Then close all plugins.
  for (auto const& plugin : mPlugins) {
    closePlugin(plugin.first);
  }

  mPlugins.clear();

  // Then unload SPICE.
  mSolarSystem->deinit();

  // Make sure all shared pointers have been cleared nicely. Print a warning if some references are
  // still hanging around.
  mDragNavigation.reset();

  auto assertCleanUp = [](std::string const& name, size_t count) {
    if (count > 1) {
      spdlog::warn(
          "Failed to properly cleanup the Application: Use count of '{}' is {} but should be 0.",
          name, count - 1);
    }
  };

  unregisterGuiCallbacks();

  assertCleanUp("mSolarSystem", mSolarSystem.use_count());
  mSolarSystem.reset();

  assertCleanUp("mTimeControl", mTimeControl.use_count());
  mTimeControl.reset();

  assertCleanUp("mGuiManager", mGuiManager.use_count());
  mGuiManager.reset();

  assertCleanUp("mGraphicsEngine", mGraphicsEngine.use_count());
  mGraphicsEngine.reset();

  assertCleanUp("mFrameTimings", mFrameTimings.use_count());
  mFrameTimings.reset();

  assertCleanUp("mInputManager", mInputManager.use_count());
  mInputManager.reset();

  assertCleanUp("mSettings", mSettings.use_count());
  mSettings.reset();

  VistaFrameLoop::Quit();
}

////////////////////////////////////////////////////////////////////////////////////////////////////

void Application::FrameUpdate() {

  // The FrameTimings are used to measure the time individual parts of the frame loop require.
  mFrameTimings->startFullFrameTiming();

  // Increase the frame count once every frame.
  ++m_iFrameCount;

  // At the beginning of each frame, the slaves (if any) are synchronized with the master.
  {
    cs::utils::FrameTimings::ScopedTimer timer("ClusterMode StartFrame");
    m_pClusterMode->StartFrame();
  }

  // Emit vista events.
  if (m_pClusterMode->GetIsLeader()) {
    cs::utils::FrameTimings::ScopedTimer timer("Emit VistaSystemEvents");
    EmitSystemEvent(VistaSystemEvent::VSE_POSTGRAPHICS);
    EmitSystemEvent(VistaSystemEvent::VSE_PREAPPLICATIONLOOP);
    EmitSystemEvent(VistaSystemEvent::VSE_UPDATE_INTERACTION);
    EmitSystemEvent(VistaSystemEvent::VSE_UPDATE_DISPLAYS);
    EmitSystemEvent(VistaSystemEvent::VSE_POSTAPPLICATIONLOOP);
    EmitSystemEvent(VistaSystemEvent::VSE_UPDATE_DELAYED_INTERACTION);
    EmitSystemEvent(VistaSystemEvent::VSE_PREGRAPHICS);
  }

  // update vista classes --------------------------------------------------------------------------

  {
    cs::utils::FrameTimings::ScopedTimer timer("ClusterMode ProcessFrame");
    m_pClusterMode->ProcessFrame();
  }

  {
    cs::utils::FrameTimings::ScopedTimer timer("ClusterMode EndFrame");
    m_pClusterMode->EndFrame();
  }

  {
    cs::utils::FrameTimings::ScopedTimer timer("AvgLoopTime RecordTime");
    m_pAvgLoopTime->RecordTime();
  }

  // hot-reloading of plugins ----------------------------------------------------------------------

  for (auto const& plugin : mPluginsToUnload) {
    deinitPlugin(plugin);
    closePlugin(plugin);
  }
  mPluginsToUnload.clear();

  for (auto const& plugin : mPluginsToLoad) {
    openPlugin(plugin);
    initPlugin(plugin);
  }
  mPluginsToLoad.clear();

  // download datsets at application startup -------------------------------------------------------

  // At frame 25 we start to download datasets. This ensures that the loading screen is actually
  // already visible.
  if (GetFrameCount() == 25) {
    if (mSettings->mDownloadData.size() > 0) {
      // Download datasets in parallel. We use 10 threads to download the data.
      mDownloader.reset(new cs::utils::Downloader(10));
      for (auto const& download : mSettings->mDownloadData) {
        mDownloader->download(download.mUrl, download.mFile);
      }

      // If all files were already downloaded, this could have gone quite quickly...
      if (mDownloader->hasFinished()) {
        mDownloadedData = true;
        mDownloader.release();
      } else {
        // Show to the user what's going on.
        mGuiManager->setLoadingScreenStatus("Downloading data...");
      }

    } else {
      // There are actually no datasets to download, so we can just set mDownloadedData to true.
      mDownloadedData = true;
    }
  }

  // Until everything is downloaded, update the progressbar accordingly.
  if (!mDownloadedData && mDownloader) {
    mGuiManager->setLoadingScreenProgress(static_cast<float>(mDownloader->getProgress()), false);
  }

  // Once the data download has finished, we can delete our downloader.
  if (!mDownloadedData && mDownloader && mDownloader->hasFinished()) {
    mDownloadedData = true;
    mDownloader.release();
  }

  // If all data is available, we can initialize the SolarSystem. This can only be done after the
  // data download, as it requires SPICE kernels which might be part of the download.
  if (mDownloadedData && !mSolarSystem->getIsInitialized()) {
    try {
      mSolarSystem->init(mSettings->mSpiceKernel);
    } catch (std::runtime_error const& e) {
      spdlog::error("Failed to initialize the SolarSystem: {}", e.what());
      Quit();
    }

    // Store the frame at which we should start loading the plugins.
    mStartPluginLoadingAtFrame = GetFrameCount();
  }

  // load plugins at application startup -----------------------------------------------------------

  // Once all data has been downloaded and the SolarSystem has been initialized, we can start
  // loading the plugins.
  if (mDownloadedData && !mLoadedAllPlugins) {

    // Before loading the first plugin and between loading the individual plugins, we will draw some
    // frames. This allows the loading screen to update the status message and move the progress
    // bar. For now, we wait a hard-coded number of 25 frames before and between loading of the
    // plugins.
    const int32_t cLoadingDelay = 25;

    // Every 25th frame something happens. At frame X we will show the name of the plugin on the
    // loading screen which will be loaded at frame X+25. At frame X+25 we will load the according
    // plugin and also update the loading screen to display the name of the plugin which is going to
    // be loaded at fram X+50. And so on.
    if (((GetFrameCount() - mStartPluginLoadingAtFrame) % cLoadingDelay) == 0) {

      // Calculate the index of the plugin which should be loaded this frame.
      int32_t pluginToLoad = (GetFrameCount() - mStartPluginLoadingAtFrame) / cLoadingDelay - 1;

      if (pluginToLoad >= 0 && pluginToLoad < static_cast<int32_t>(mPlugins.size())) {

        // Get an iterator pointing to the plugin handle.
        auto plugin = mPlugins.begin();
        std::advance(plugin, pluginToLoad);

        initPlugin(plugin->first);

      } else if (pluginToLoad == static_cast<int32_t>(mPlugins.size())) {

        spdlog::info("Ready for Takeoff!");

        // Once all plugins have been loaded, we set a boolean indicating this state.
        mLoadedAllPlugins = true;

        // Update the loading screen status.
        mGuiManager->setLoadingScreenStatus("Ready for Takeoff");
        mGuiManager->setLoadingScreenProgress(100.f, true);

        // We will keep the loading screen active for some frames, as the first frames are usually a
        // bit choppy as data is uploaded to the GPU.
        mHideLoadingScreenAtFrame = GetFrameCount() + cLoadingDelay;

        // initial observer animation --------------------------------------------------------------

        // At application startup, the celestial observer is transitioned to its position specified
        // in the setting json file.
        auto const& observerSettings = mSettings->mObserver;
        glm::dvec2  lonLat(observerSettings.mLongitude, observerSettings.mLatitude);
        lonLat = cs::utils::convert::toRadians(lonLat);

        auto radii = cs::core::SolarSystem::getRadii(observerSettings.mCenter);

        if (radii[0] == 0.0 || radii[2] == 0.0) {
          radii = glm::dvec3(1, 1, 1);
        }

        // Multiply longitude and latitude of start location by 0.5 to create more interesting start
        // animation.
        auto cart = cs::utils::convert::toCartesian(
            lonLat * 0.5, radii[0], radii[0], observerSettings.mDistance * 5);

        glm::dvec3 y = glm::dvec3(0, -1, 0);
        glm::dvec3 z = cart;
        glm::dvec3 x = glm::cross(z, y);
        y            = glm::cross(z, x);

        x = glm::normalize(x);
        y = glm::normalize(y);
        z = glm::normalize(z);

        auto rotation = glm::toQuat(glm::dmat3(x, y, z));
        mSolarSystem->getObserver().setCenterName(observerSettings.mCenter);
        mSolarSystem->getObserver().setFrameName(observerSettings.mFrame);
        mSolarSystem->getObserver().setAnchorPosition(cart);
        mSolarSystem->getObserver().setAnchorRotation(rotation);

        mSolarSystem->flyObserverTo(observerSettings.mCenter, observerSettings.mFrame, lonLat,
            observerSettings.mDistance, 5.0);
      }

      // If there is a plugin going to be loaded after the next cLoadingDelay frames, display its
      // name on the loading screen and update the progress accordingly.
      if (pluginToLoad + 1 < static_cast<int32_t>(mPlugins.size())) {
        auto plugin = mPlugins.begin();
        std::advance(plugin, pluginToLoad + 1);
        mGuiManager->setLoadingScreenStatus("Loading " + plugin->first + " ...");
        mGuiManager->setLoadingScreenProgress(100.f * (pluginToLoad + 1) / mPlugins.size(), true);
      }
    }
  }

  // Main classes are only updated once all plugins have been loaded.
  if (mLoadedAllPlugins) {

    // Hide the loading screen after several frames.
    if (GetFrameCount() == mHideLoadingScreenAtFrame) {
      mGuiManager->enableLoadingScreen(false);
    }

    // update CosmoScout VR classes ----------------------------------------------------------------

    // Update the InputManager.
    {
      cs::utils::FrameTimings::ScopedTimer timer(
          "InputManager Update", cs::utils::FrameTimings::QueryMode::eCPU);
      mInputManager->update();
    }

    // Update the TimeControl.
    {
      cs::utils::FrameTimings::ScopedTimer timer(
          "TimeControl Update", cs::utils::FrameTimings::QueryMode::eCPU);
      mTimeControl->update();
    }

    // Update the navigation, SolarSystem and scene scale.
    {
      cs::utils::FrameTimings::ScopedTimer timer(
          "SolarSystem Update", cs::utils::FrameTimings::QueryMode::eCPU);
      mDragNavigation->update();
      mSolarSystem->update();
      mSolarSystem->updateSceneScale();
      mSolarSystem->updateObserverFrame();
    }

    // Update the individual plugins.
    for (auto const& plugin : mPlugins) {
      cs::utils::FrameTimings::ScopedTimer timer(
          plugin.first, cs::utils::FrameTimings::QueryMode::eBoth);

      try {
        plugin.second.mPlugin->update();
      } catch (std::runtime_error const& e) {
        spdlog::error("Error updating plugin '{}': {}", plugin.first, e.what());
      }
    }

    // Synchronize the observer position and simulation time across the network.
    {
      cs::utils::FrameTimings::ScopedTimer timer(
          "Scene Sync", cs::utils::FrameTimings::QueryMode::eCPU);

      struct SyncMessage {
        glm::dvec3 mPosition;
        glm::dquat mRotation;
        double     mScale;
        double     mTime;
      } syncMessage;

      syncMessage.mPosition = mSolarSystem->getObserver().getAnchorPosition();
      syncMessage.mRotation = mSolarSystem->getObserver().getAnchorRotation();
      syncMessage.mScale    = mSolarSystem->getObserver().getAnchorScale();
      syncMessage.mTime     = mTimeControl->pSimulationTime.get();

      std::string frame(mSolarSystem->getObserver().getFrameName());
      std::string center(mSolarSystem->getObserver().getCenterName());

      {
        std::vector<VistaType::byte> data(sizeof(SyncMessage));
        std::memcpy(&data[0], &syncMessage, sizeof(SyncMessage));
        mSceneSync->SyncData(data);
        std::memcpy(&syncMessage, &data[0], sizeof(SyncMessage));
      }

      mSceneSync->SyncData(frame);
      mSceneSync->SyncData(center);

      mSolarSystem->getObserver().setFrameName(frame);
      mSolarSystem->getObserver().setCenterName(center);

      mSolarSystem->getObserver().setAnchorPosition(syncMessage.mPosition);
      mSolarSystem->getObserver().setAnchorRotation(syncMessage.mRotation);
      mSolarSystem->getObserver().setAnchorScale(syncMessage.mScale);

      mTimeControl->pSimulationTime = syncMessage.mTime;
    }

    // Update the GraphicsEngine.
    {
      auto sunTransform = mSolarSystem->getSun()->getWorldTransform();
      mGraphicsEngine->update(glm::normalize(sunTransform[3].xyz()));
    }
  }

  // Update the user interface.
  {
    cs::utils::FrameTimings::ScopedTimer timer("User Interface");

    // Call update on all APIs
    mGuiManager->getGui()->callJavascript("CosmoScout.update");

    if (mSolarSystem->pActiveBody.get()) {

      // Update the user's position display in the header bar.
      auto                pSG = GetVistaSystem()->GetGraphicsManager()->GetSceneGraph();
      VistaTransformNode* pTrans =
          dynamic_cast<VistaTransformNode*>(pSG->GetNode("Platform-User-Node"));

      auto vWorldPos = glm::vec4(1);
      pTrans->GetWorldPosition(vWorldPos.x, vWorldPos.y, vWorldPos.z);

      auto radii = mSolarSystem->pActiveBody.get()->getRadii();
      auto vPlanetPos =
          glm::inverse(mSolarSystem->pActiveBody.get()->getWorldTransform()) * vWorldPos;
      auto   polar = cs::utils::convert::toLngLatHeight(vPlanetPos.xyz(), radii[0], radii[0]);
      double surfaceHeight = mSolarSystem->pActiveBody.get()->getHeight(polar.xy());
      double heightDiff    = polar.z / mGraphicsEngine->pHeightScale.get() - surfaceHeight;

      if (!std::isnan(polar.x) && !std::isnan(polar.y) && !std::isnan(heightDiff)) {
        mGuiManager->getGui()->executeJavascript(
            fmt::format("CosmoScout.state.observerPosition = [{}, {}, {}]",
                cs::utils::convert::toDegrees(polar.x), cs::utils::convert::toDegrees(polar.y),
                heightDiff));
      }

      // Update the compass in the header bar.
      auto rot = mSolarSystem->getObserver().getRelativeRotation(
          mTimeControl->pSimulationTime.get(), *mSolarSystem->pActiveBody.get());
      glm::dvec4 up(0.0, 1.0, 0.0, 0.0);
      glm::dvec4 north = rot * up;
      north.z          = 0.0;

      double angle = std::acos(glm::dot(up, glm::normalize(north)));
      if (north.x < 0.0) {
        angle = -angle;
      }

      mGuiManager->getGui()->callJavascript("CosmoScout.timeline.setNorthDirection", angle);
    }

    mGuiManager->update();
  }

  // update vista classes --------------------------------------------------------------------------

  {
    cs::utils::FrameTimings::ScopedTimer timer("DisplayManager DrawFrame");
    m_pDisplayManager->DrawFrame();
  }

  {
    cs::utils::FrameTimings::ScopedTimer timer("ClusterMode SwapSync");
    m_pClusterMode->SwapSync();
  }

  // Measure frame time until here. If we moved this farther down, we would also measure the
  // vertical synchronization delay, which would result in wrong timings.
  mFrameTimings->endFullFrameTiming();

  {
    cs::utils::FrameTimings::ScopedTimer timer("DisplayManager DisplayFrame");
    m_pDisplayManager->DisplayFrame();
  }

  {
    cs::utils::FrameTimings::ScopedTimer timer("FrameRate RecordTime");
    m_pFrameRate->RecordTime();
  }

  // Record frame timings.
  mFrameTimings->update();
}

////////////////////////////////////////////////////////////////////////////////////////////////////

void Application::testLoadAllPlugins() {

  auto plugins = cs::utils::filesystem::listFiles(PLUGIN_PATH);

  for (auto const& plugin : plugins) {
    if (cs::utils::endsWith(plugin, ".so") || cs::utils::endsWith(plugin, ".dll")) {
      // Clear errors.
      LIBERROR();

      COSMOSCOUT_LIBTYPE pluginHandle = OPENLIB(plugin.c_str());

      if (pluginHandle) {
        cs::core::PluginBase* (*pluginConstructor)();
        pluginConstructor = (cs::core::PluginBase * (*)()) LIBFUNC(pluginHandle, "create");

        if (pluginConstructor) {
          spdlog::info("Plugin '{}' found.", plugin);
        } else {
          spdlog::error("Failed to load plugin '{}': Plugin has no 'create' method.", plugin);
        }

      } else {
        spdlog::error("Failed to load plugin '{}': {}", plugin, LIBERROR());
      }
    }
  }
}

////////////////////////////////////////////////////////////////////////////////////////////////////

void Application::openPlugin(std::string const& name) {
  auto plugin = mPlugins.find(name);

  if (plugin == mPlugins.end()) {
    try {

#ifdef __linux__
      std::string path = PLUGIN_PATH "lib" + name + ".so";
#else
      std::string path = PLUGIN_PATH + name + ".dll";
#endif

      // Clear errors.
      LIBERROR();

      COSMOSCOUT_LIBTYPE pluginHandle = OPENLIB(path.c_str());

      if (pluginHandle) {
        cs::core::PluginBase* (*pluginConstructor)();
        pluginConstructor = (cs::core::PluginBase * (*)()) LIBFUNC(pluginHandle, "create");

        spdlog::info("Opening plugin '{}'.", name);

        // Actually call the plugin's constructor and add the returned pointer to out list.
        mPlugins.insert(std::pair<std::string, Plugin>(name, {pluginHandle, pluginConstructor()}));
      } else {
        spdlog::error("Failed to load plugin '{}': {}", name, LIBERROR());
      }
    } catch (std::exception const& e) {
      spdlog::error("Failed to load plugin '{}': {}", name, e.what());
    }
  } else {
    spdlog::warn("Cannot open plugin '{}': Plugin is already opened!", name);
  }
}

////////////////////////////////////////////////////////////////////////////////////////////////////

void Application::initPlugin(std::string const& name) {
  auto plugin = mPlugins.find(name);

  if (plugin != mPlugins.end()) {
    if (!plugin->second.mIsInitialized) {

      // First provide the plugin with all required class instances.
      plugin->second.mPlugin->setAPI(mSettings, mSolarSystem, mGuiManager, mInputManager,
          GetVistaSystem()->GetGraphicsManager()->GetSceneGraph(), mGraphicsEngine, mFrameTimings,
          mTimeControl);

      // Then do the actual initialization. This may actually take a while and the application
      // will become unresponsive in the meantime.
      try {
        plugin->second.mPlugin->init();
        plugin->second.mIsInitialized = true;

        // Plugin finished loading -> init its custom components.
        mGuiManager->getGui()->callJavascript("CosmoScout.gui.initInputs");
      } catch (std::exception const& e) {
        spdlog::error("Failed to initialize plugin '{}': {}", plugin->first, e.what());
      }
    } else {
      spdlog::warn("Cannot initialize plugin '{}': Plugin is already initialized!", name);
    }
  }
}

////////////////////////////////////////////////////////////////////////////////////////////////////

void Application::deinitPlugin(std::string const& name) {
  auto plugin = mPlugins.find(name);

  if (plugin != mPlugins.end()) {
    if (plugin->second.mIsInitialized) {
      plugin->second.mPlugin->deInit();
      plugin->second.mIsInitialized = false;
    } else {
      spdlog::warn("Cannot deinitialize plugin '{}': Plugin is not initialized!", name);
    }
  } else {
    spdlog::warn("Cannot unload plugin '{}': No plugin loaded with this name!", name);
  }
}

////////////////////////////////////////////////////////////////////////////////////////////////////

void Application::closePlugin(std::string const& name) {
  auto plugin = mPlugins.find(name);

  if (plugin != mPlugins.end()) {
    spdlog::info("Closing plugin '{}'.", plugin->first);

    auto handle           = plugin->second.mHandle;
    auto pluginDestructor = (void (*)(cs::core::PluginBase*))LIBFUNC(handle, "destroy");

    pluginDestructor(plugin->second.mPlugin);
    CLOSELIB(handle);

    mPlugins.erase(plugin);

  } else {
    spdlog::warn("Failed to close plugin '{}': No plugin loaded with this name!", name);
  }
}

////////////////////////////////////////////////////////////////////////////////////////////////////

void Application::connectSlots() {

  // Update mouse pointer coordinate display in the user interface.
  mInputManager->pHoveredObject.connect([this](cs::core::InputManager::Intersection intersection) {
    if (intersection.mObject) {
      auto body = std::dynamic_pointer_cast<cs::scene::CelestialBody>(intersection.mObject);

      if (body) {
        auto radii = body->getRadii();
        auto polar = cs::utils::convert::toLngLatHeight(intersection.mPosition, radii[0], radii[0]);
        auto lngLat = cs::utils::convert::toDegrees(polar.xy());

        if (!std::isnan(lngLat.x) && !std::isnan(lngLat.y) && !std::isnan(polar.z)) {
          mGuiManager->getGui()->executeJavascript(
              fmt::format("CosmoScout.state.pointerPosition = [{}, {}, {}];", lngLat.x, lngLat.y,
                  polar.z / mGraphicsEngine->pHeightScale.get()));
          return;
        }
      }
    }
    mGuiManager->getGui()->executeJavascript("CosmoScout.state.pointerPosition = undefined;");
  });

  // Update the time shown in the user interface when the simulation time changes.
  mTimeControl->pSimulationTime.connect([this](double val) {
    std::stringstream sstr;
    auto              facet = new boost::posix_time::time_facet();
    facet->format("%d-%b-%Y %H:%M:%S.%f");
    sstr.imbue(std::locale(std::locale::classic(), facet));
    sstr << cs::utils::convert::toBoostTime(val);
    mGuiManager->getGui()->callJavascript("CosmoScout.timeline.setDate", sstr.str());
  });

  // Update the simulation time speed shown in the user interface.
  mTimeControl->pTimeSpeed.connect([this](float val) {
    mGuiManager->getGui()->callJavascript("CosmoScout.timeline.setTimeSpeed", val);
  });

  // Show notification when the center name of the celestial observer changes.
  mSolarSystem->pObserverCenter.connect([this](std::string const& center) {
    if (center == "Solar System Barycenter") {
      mGuiManager->showNotification("Leaving " + mSolarSystem->pActiveBody.get()->getCenterName(),
          "Now travelling in free space.", "star");
    } else {
      mGuiManager->showNotification(
          "Approaching " + mSolarSystem->pActiveBody.get()->getCenterName(),
          "Position is locked to " + mSolarSystem->pActiveBody.get()->getCenterName() + ".",
          "public");
    }
    mGuiManager->getGui()->executeJavascript(
        fmt::format("CosmoScout.state.activePlanetCenter = '{}';", center));
  });

  // Show notification when the frame name of the celestial observer changes.
  mSolarSystem->pObserverFrame.connect([this](std::string const& frame) {
    if (frame == "J2000") {
      mGuiManager->showNotification(
          "Stop tracking " + mSolarSystem->pActiveBody.get()->getCenterName(),
          "Orbit is not synced anymore.", "vpn_lock");
    } else {
      mGuiManager->showNotification("Tracking " + mSolarSystem->pActiveBody.get()->getCenterName(),
          "Orbit in sync with " + mSolarSystem->pActiveBody.get()->getCenterName() + ".",
          "vpn_lock");
    }
    mGuiManager->getGui()->executeJavascript(
        fmt::format("CosmoScout.state.activePlanetFrame = '{}';", frame));
  });

  // Show the current speed of the celestial observer in the user interface.
  mSolarSystem->pCurrentObserverSpeed.connect([this](float speed) {
    mGuiManager->getGui()->executeJavascript(
        fmt::format("CosmoScout.state.observerSpeed = {};", speed));
  });

  // Show the statistics GuiItem when measurements are enabled.
  mFrameTimings->pEnableMeasurements.connect(
      [this](bool enable) { mGuiManager->getStatistics()->setIsEnabled(enable); });

  mOnMessageConnection = cs::utils::logger::onMessage().connect(
      [this](
          std::string const& logger, spdlog::level::level_enum level, std::string const& message) {
        const std::unordered_map<spdlog::level::level_enum, std::string> mapping = {
            {spdlog::level::trace, "T"}, {spdlog::level::debug, "D"}, {spdlog::level::info, "I"},
            {spdlog::level::warn, "W"}, {spdlog::level::err, "E"}, {spdlog::level::critical, "C"}};

        mGuiManager->getGui()->callJavascript(
            "CosmoScout.statusbar.printMessage", mapping.at(level), logger, message);
      });
}

////////////////////////////////////////////////////////////////////////////////////////////////////

void Application::registerGuiCallbacks() {

  // Unloads a plugin.
  mGuiManager->getGui()->registerCallback("core.unloadPlugin",
      "Unloads the plugin with the given name.", std::function([this](std::string&& pluginName) {
        // We do not directly unload the plugin, as this callback is triggered from the
        // GuiManager->update(). Doing this here could lead to deadlocks.
        mPluginsToUnload.insert(pluginName);
      }));

  // Loads a plugin.
  mGuiManager->getGui()->registerCallback("core.loadPlugin",
      "Loads the plugin with the given name.", std::function([this](std::string&& pluginName) {
        // We do not directly load the plugin, as this callback is triggered from the
        // GuiManager->update(). Doing this here could lead to deadlocks.
        mPluginsToLoad.insert(pluginName);
      }));

  // Reloads a plugin.
  mGuiManager->getGui()->registerCallback("core.reloadPlugin",
      "Reloads the plugin with the given name.", std::function([this](std::string&& pluginName) {
        // We do not directly reload the plugin, as this callback is triggered from the
        // GuiManager->update(). Doing this here could lead to deadlocks.
        mPluginsToUnload.insert(pluginName);
        mPluginsToLoad.insert(pluginName);
      }));

  // Lists all loaded plugins.
  mGuiManager->getGui()->registerCallback(
      "core.listPlugins", "Lists all loaded plugins.", std::function([this]() {
        for (auto const& plugin : mPlugins) {
          spdlog::info(plugin.first);
        }
      }));

  // graphics callbacks ----------------------------------------------------------------------------

  // Enables lighting computation globally.
  mGuiManager->getGui()->registerCallback("graphics.setEnableLighting",
      "Enables or disables lighting computations for planet surfaces.",
      std::function([this](bool enable) { mGraphicsEngine->pEnableLighting = enable; }));

  // Shows cascaded shadow mapping debugging information on the terrain.
  mGuiManager->getGui()->registerCallback("graphics.setEnableCascadesDebug",
      "Enables or disables a debug visualization for the shadow maps.",
      std::function([this](bool enable) { mGraphicsEngine->pEnableShadowsDebug = enable; }));

  // Enables the calculation of shadows.
  mGuiManager->getGui()->registerCallback("graphics.setEnableShadows",
      "Enables or disables calculation of shadow maps.",
      std::function([this](bool enable) { mGraphicsEngine->pEnableShadows = enable; }));

  // Freezes the shadow frustum.
  mGuiManager->getGui()->registerCallback("graphics.setEnableShadowFreeze",
      "If enabled, the camera frustum used for the calculation of the shadow map cascades is not "
      "updated anymore.",
      std::function([this](bool enable) { mGraphicsEngine->pEnableShadowsFreeze = enable; }));

  // Sets a value which individual plugins may honor trading rendering fidelity for performance.
  mGuiManager->getGui()->registerCallback("graphics.setLightingQuality",
      "Sets the quality for lighting computations. This can be either 0, 1 or 2.",
      std::function(
          [this](double value) { mGraphicsEngine->pLightingQuality = static_cast<int>(value); }));

  // Adjusts the resolution of the shadowmap.
  mGuiManager->getGui()->registerCallback("graphics.setShadowmapResolution",
      "Sets the resolution of the shadow maps. This should be a power of two, e.g. 256, 512, 1024, "
      "etc.",
      std::function(
          [this](double val) { mGraphicsEngine->pShadowMapResolution = static_cast<int>(val); }));

  // Adjusts the number of shadowmap cascades.
  mGuiManager->getGui()->registerCallback("graphics.setShadowmapCascades",
      "Sets the number of shadow map cascades. Should be in the range of 1-5.",
      std::function(
          [this](double val) { mGraphicsEngine->pShadowMapCascades = static_cast<int>(val); }));

  // Adjusts the depth range of the shadowmap.
  mGuiManager->getGui()->registerCallback("graphics.setShadowmapRange",
      "Sets one end of the shadow distance range. The first parameter is the actual value in "
      "viewspace, the second specifies which end to set: Zero for the closer end; One for the "
      "farther end.",
      std::function([this](double val, double handle) {
        glm::vec2 range = mGraphicsEngine->pShadowMapRange.get();

        if (handle == 0.0) {
          range.x = (float)val;
        } else {
          range.y = (float)val;
        };

        mGraphicsEngine->pShadowMapRange = range;
      }));

  // Adjusts the additional frustum length for shadowmap rendering in sun space.
  mGuiManager->getGui()->registerCallback("graphics.setShadowmapExtension",
      "Sets one end of the shadow frustum range in sun direction. The first parameter is the "
      "actual value in sunspace, the second specifies which end to set: Zero for the closer end; "
      "One for the farther end.",
      std::function([this](double val, double handle) {
        glm::vec2 extension = mGraphicsEngine->pShadowMapExtension.get();

        if (handle == 0.0) {
          extension.x = (float)val;
        } else {
          extension.y = (float)val;
        };

        mGraphicsEngine->pShadowMapExtension = extension;
      }));

  // Adjusts the distribution of shadowmap cascades.
  mGuiManager->getGui()->registerCallback("graphics.setShadowmapSplitDistribution",
      "Defines an exponent for the distribution of the shadowmap cascades.",
      std::function([this](double val) {
        mGraphicsEngine->pShadowMapSplitDistribution = static_cast<float>(val);
      }));

  // Adjusts the bias to mitigate shadow acne.
  mGuiManager->getGui()->registerCallback("graphics.setShadowmapBias",
      "Sets the bias for the shadow map lookups.", std::function([this](double val) {
        mGraphicsEngine->pShadowMapBias = static_cast<float>(val);
      }));

  // A global factor which plugins may honor when they render some sort of terrain.
  mGuiManager->getGui()->registerCallback("graphics.setTerrainHeight",
      "Sets a factor for the height exaggeration of the planet's surface.",
      std::function(
          [this](double value) { mGraphicsEngine->pHeightScale = static_cast<float>(value); }));

  // Adjusts the global scaling of world-space widgets.
  mGuiManager->getGui()->registerCallback("graphics.setWidgetScale",
      "Sets a factor for the scaling of world space user interface elements.",
      std::function(
          [this](double value) { mGraphicsEngine->pWidgetScale = static_cast<float>(value); }));

  // Adjusts the sensor diagonal of the virtual camera.
  mGuiManager->getGui()->registerCallback("graphics.setSensorDiagonal",
      "Sets the sensor diagonal of the virtual camera in [mm].", std::function([this](double val) {
        mGraphicsEngine->pSensorDiagonal = static_cast<float>(val);
      }));

  // Adjusts the foacl length of the virtual camera.
  mGuiManager->getGui()->registerCallback("graphics.setFocalLength",
      "Sets the focal length of the virtual camera in [mm].", std::function([this](double val) {
        mGraphicsEngine->pFocalLength = static_cast<float>(val);
      }));

  // Toggles HDR rendering.
  mGuiManager->getGui()->registerCallback("graphics.setEnableHDR",
      "Enables or disables HDR rendering.",
      std::function([this](bool val) { mGraphicsEngine->pEnableHDR = val; }));

  // Toggles auto-exposure.
  mGuiManager->getGui()->registerCallback("graphics.setEnableAutoExposure",
      "Enables or disables automatic exposure calculation.",
      std::function([this](bool val) { mGraphicsEngine->pEnableAutoExposure = val; }));

  // Adjusts the exposure compensation for HDR rendering.
  mGuiManager->getGui()->registerCallback("graphics.setExposureCompensation",
      "Adds some additional exposure in [EV].", std::function([this](double val) {
        mGraphicsEngine->pExposureCompensation = static_cast<float>(val);
      }));

  // Adjusts the exposure of the virtual camera in HDR mode.
  mGuiManager->getGui()->registerCallback("graphics.setExposure",
      "Sets the exposure of the image in [EV]. Only available if auto-exposure is disabled.",
      std::function([this](double val) {
        if (!mGraphicsEngine->pEnableAutoExposure.get()) {
          mGraphicsEngine->pExposure = static_cast<float>(val);
        }
      }));

  // Adjusts how fast the exposure adapts to new lighting condtitions.
  mGuiManager->getGui()->registerCallback("graphics.setExposureAdaptionSpeed",
      "Adjust the quickness of auto-exposure.", std::function([this](double val) {
        mGraphicsEngine->pExposureAdaptionSpeed = static_cast<float>(val);
      }));

  // If auto-exposure is enabled, we update the slider in the user interface to show the current
  // value.
  mGraphicsEngine->pExposure.connect([this](float value) {
    if (mGraphicsEngine->pEnableAutoExposure.get()) {
      mGuiManager->getGui()->callJavascript(
          "CosmoScout.gui.setSliderValue", "graphics.setExposure", value);
    }
  });

  // Toggles auto-glow.
  mGuiManager->getGui()->registerCallback("graphics.setEnableAutoGlow",
      "If enabled, the glow amount is chosen based on the current exposure.",
      std::function([this](bool val) { mGraphicsEngine->pEnableAutoGlow = val; }));

  // If auto-glow is enabled, we update the slider in the user interface to show the current value.
  mGraphicsEngine->pGlowIntensity.connect([this](float value) {
    if (mGraphicsEngine->pEnableAutoGlow.get()) {
      mGuiManager->getGui()->callJavascript(
          "CosmoScout.gui.setSliderValue", "graphics.setGlowIntensity", value);
    }
  });

  // Update the side bar field showing the average luminance of the scene.
  mGraphicsEngine->pAverageLuminance.connect([this](float value) {
    mGuiManager->getGui()->callJavascript("CosmoScout.sidebar.setAverageSceneLuminance", value);
  });

  // Update the side bar field showing the maximum luminance of the scene.
  mGraphicsEngine->pMaximumLuminance.connect([this](float value) {
    mGuiManager->getGui()->callJavascript("CosmoScout.sidebar.setMaximumSceneLuminance", value);
  });

  // Adjusts the amount of ambient lighting.
  mGuiManager->getGui()->registerCallback("graphics.setAmbientLight",
      "Sets the amount of ambient light.", std::function([this](double val) {
        mGraphicsEngine->pAmbientBrightness = static_cast<float>(std::pow(val, 10.0));
      }));

  // Adjusts the amount of artificial glare in HDR mode.
  mGuiManager->getGui()->registerCallback("graphics.setGlowIntensity",
      "Adjusts the amount of glow of overexposed areas.", std::function([this](double val) {
        mGraphicsEngine->pGlowIntensity = static_cast<float>(val);
      }));

  // Adjusts the exposure range for auto exposure.
  mGuiManager->getGui()->registerCallback("graphics.setExposureRange",
      "Sets the minimum and maximum value for auto-exposure. The first paramater is the actual "
      "value in [EV], the second determines which to sets: Zero for the lower end; one for the "
      "upper end.",
      std::function([this](double val, double handle) {
        glm::vec2 range = mGraphicsEngine->pAutoExposureRange.get();

        if (handle == 0.0)
          range.x = static_cast<float>(val);
        else
          range.y = static_cast<float>(val);

        mGraphicsEngine->pAutoExposureRange = range;
      }));

  // Enables or disables the per-frame time measurements.
  mGuiManager->getGui()->registerCallback("graphics.setEnableTimerQueries",
      "Shows or hides the frame timing information.",
      std::function([this](bool value) { mFrameTimings->pEnableMeasurements = value; }));

  // Enables or disables vertical synchronization.
  mGuiManager->getGui()->registerCallback("graphics.setEnableVsync",
      "Enables or disables vertical synchronization.", std::function([](bool value) {
        GetVistaSystem()
            ->GetDisplayManager()
            ->GetWindows()
            .begin()
            ->second->GetWindowProperties()
            ->SetVSyncEnabled(value);
      }));

  // Timeline callbacks ----------------------------------------------------------------------------

  // Sets the current simulation time. The argument must be a string accepted by
  // TimeControl::setTime.
  mGuiManager->getGui()->registerCallback("time.setDate",
      "Sets the current simulation time. Format must be in the format '2002-01-20 "
      "23:59:59.000'.",
      std::function([this](std::string&& sDate) {
        double time = cs::utils::convert::toSpiceTime(boost::posix_time::time_from_string(sDate));
        mTimeControl->setTime(time);
      }));

  // Sets the current simulation time. The argument must be a double representing Barycentric
  // Dynamical Time.
  mGuiManager->getGui()->registerCallback("time.set",
      "Sets the current simulation time. The value must be in barycentric dynamical time. If the "
      "absolute difference to the current simulation time is lower than the given threshold "
      "(optionalDouble2, default is 172800s which is 48h), there will be a transition of the given "
      "duration (optionalDouble, default is 0s).",
      std::function(
          [this](double tTime, std::optional<double> duration, std::optional<double> threshold) {
            mTimeControl->setTime(tTime, duration.value_or(0.0), threshold.value_or(48 * 60 * 60));
          }));

  // Resets the time to the configured start time.
  mGuiManager->getGui()->registerCallback("time.reset",
      "Resets the simulation time to the default value. If the absolute difference to the current "
      "simulation time is lower than the given threshold (optionalDouble2, default is 172800s "
      "which is 48h), there will be a transition of the given duration (optionalDouble, default is "
      "0s).",
      std::function([this](std::optional<double> duration, std::optional<double> threshold) {
        mTimeControl->resetTime(duration.value_or(0.0), threshold.value_or(48 * 60 * 60));
      }));

  // Modifies the current simulation time by adding some (fractional) hours.
  mGuiManager->getGui()->registerCallback("time.addHours",
      "Adds the given amount of hours to the current simulation time. If the amount is lower than "
      "the given threshold (optionalDouble2, default is 172800s which is 48h), there will be a "
      "transition of the given duration (optionalDouble, default is 0s).",
      std::function(
          [this](double amount, std::optional<double> duration, std::optional<double> threshold) {
            mTimeControl->setTime(mTimeControl->pSimulationTime.get() + 60.0 * 60.0 * amount,
                duration.value_or(0.0), threshold.value_or(48 * 60 * 60));
          }));

  // Adjusts the simulation time speed.
  mGuiManager->getGui()->registerCallback("time.setSpeed",
      "Sets the multiplier for the simulation time speed.", std::function([this](double speed) {
        mTimeControl->setTimeSpeed(static_cast<float>(speed));
      }));

  // navigation callbacks --------------------------------------------------------------------------

  // Sets the observer position to the given cartesian coordinates.
  mGuiManager->getGui()->registerCallback("navigation.setPosition",
      "Sets the observer position to the given cartesian coordinates. The optional double argument "
      "specifies the transition time in seconds (default is 5s).",
      std::function([this](double x, double y, double z, std::optional<double> duration) {
        mSolarSystem->flyObserverTo(mSolarSystem->getObserver().getCenterName(),
            mSolarSystem->getObserver().getFrameName(), glm::dvec3(x, y, z),
            mSolarSystem->getObserver().getAnchorRotation(), duration.value_or(5.0));
      }));

  // Sets the observer rotation to the given quaternion coordinates.
  mGuiManager->getGui()->registerCallback("navigation.setRotation",
      "Sets the observer rotation to the given quaternion. The optional double argument specifies "
      "the transition time in seconds (default is 2s).",
      std::function([this](double w, double x, double y, double z, std::optional<double> duration) {
        mSolarSystem->flyObserverTo(mSolarSystem->getObserver().getCenterName(),
            mSolarSystem->getObserver().getFrameName(),
            mSolarSystem->getObserver().getAnchorPosition(), glm::dquat(w, x, y, z),
            duration.value_or(2.0));
      }));

  // Flies the observer to the given celestial body.
  mGuiManager->getGui()->registerCallback("navigation.setBody",
      "Makes the observer fly to the celestial body with the given name. The optional argument "
      "specifies the travel time in seconds (default is 10s).",
      std::function([this](std::string&& name, std::optional<double> duration) {
        for (auto const& body : mSolarSystem->getBodies()) {
          if (body->getCenterName() == name) {
            mSolarSystem->flyObserverTo(
                body->getCenterName(), body->getFrameName(), duration.value_or(10.0));
            mGuiManager->showNotification("Travelling", "to " + name, "send");
            break;
          }
        }
      }));

  // Flies the celestial observer to the given location in space.
  mGuiManager->getGui()->registerCallback("navigation.setBodyLongLatHeightDuration",
      "Makes the observer fly to a given postion in space. First parameter is the target bodies "
      "name, then latitude, longitude and elevation are required. The optional double argument "
      "specifies the transition time in seconds (default is 10s).",
      std::function([this](std::string&& name, double longitude, double latitude, double height,
                        std::optional<double> duration) {
        for (auto const& body : mSolarSystem->getBodies()) {
          if (body->getCenterName() == name) {
            mSolarSystem->pActiveBody = body;
            mSolarSystem->flyObserverTo(body->getCenterName(), body->getFrameName(),
                cs::utils::convert::toRadians(glm::dvec2(longitude, latitude)), height,
                duration.value_or(10.0));
          }
        }
      }));

  // Rotates the scene in such a way, that the y-axis points towards the north pole of the currently
  // active celestial body.
  mGuiManager->getGui()->registerCallback("navigation.northUp",
      "Turns the observer so that north is facing upwards. The optional argument specifies the "
      "animation time in seconds (default is 1s).",
      std::function([this](std::optional<double> duration) {
        auto observerPos = mSolarSystem->getObserver().getAnchorPosition();

        glm::dvec3 y = glm::vec3(0, -1, 0);
        glm::dvec3 z = observerPos;
        glm::dvec3 x = glm::cross(z, y);
        y            = glm::cross(z, x);

        x = glm::normalize(x);
        y = glm::normalize(y);
        z = glm::normalize(z);

        auto rotation = glm::toQuat(glm::dmat3(x, y, z));

        mSolarSystem->flyObserverTo(mSolarSystem->getObserver().getCenterName(),
            mSolarSystem->getObserver().getFrameName(), observerPos, rotation,
            duration.value_or(1.0));
      }));

  // Rotates the scene in such a way, that the currently visible horizon is levelled.
  mGuiManager->getGui()->registerCallback("navigation.fixHorizon",
      "Turns the observer so that the horizon is horizontal. The optional argument specifies the "
      "animation time in seconds (default is 1s).",
      std::function([this](std::optional<double> duration) {
        auto radii = cs::core::SolarSystem::getRadii(mSolarSystem->getObserver().getCenterName());

        if (radii[0] == 0.0) {
          radii = glm::dvec3(1, 1, 1);
        }

        auto observerPos = mSolarSystem->getObserver().getAnchorPosition();
        auto observerRot = mSolarSystem->getObserver().getAnchorRotation();

        glm::dvec3 y = observerPos;
        glm::dvec3 z = (observerRot * glm::dvec4(0, 0.1, -1, 0)).xyz();
        glm::dvec3 x = glm::cross(z, y);
        z            = glm::cross(x, y);

        x = glm::normalize(x);
        y = glm::normalize(y);
        z = glm::normalize(z);

        auto horizonAngle =
            glm::pi<double>() * 0.5 - std::asin(std::min(1.0, radii[0] / glm::length(observerPos)));

        auto tilt     = glm::angleAxis(-horizonAngle - 0.2, glm::dvec3(1, 0, 0));
        auto rotation = glm::toQuat(glm::dmat3(x, y, z)) * tilt;

        mSolarSystem->flyObserverTo(mSolarSystem->getObserver().getCenterName(),
            mSolarSystem->getObserver().getFrameName(), observerPos, rotation,
            duration.value_or(1.0));
      }));

  // Flies the celestial observer to 0.1% of its current height.
  mGuiManager->getGui()->registerCallback("navigation.toSurface",
      "Reduces the altitude of the observer significantly. The optional argument specifies the "
      "animation time in seconds (default is 3s).",
      std::function([this](std::optional<double> duration) {
        auto radii = cs::core::SolarSystem::getRadii(mSolarSystem->getObserver().getCenterName());

        if (radii[0] == 0.0 || radii[2] == 0.0) {
          radii = glm::dvec3(1, 1, 1);
        }

        auto lngLatHeight = cs::utils::convert::toLngLatHeight(
            mSolarSystem->getObserver().getAnchorPosition(), radii[0], radii[0]);

        // fly to 0.1% of current height
        double height = lngLatHeight.z * 0.001;

        // limit to at least 10% of planet radius and at most 2m
        height = glm::clamp(height, 2.0, radii[0] * 0.1);

        if (mSolarSystem->pActiveBody.get()) {
          height += mSolarSystem->pActiveBody.get()->getHeight(lngLatHeight.xy());
        }

        height *= mGraphicsEngine->pHeightScale.get();

        auto observerPos =
            cs::utils::convert::toCartesian(lngLatHeight.xy(), radii[0], radii[0], height);
        auto observerRot = mSolarSystem->getObserver().getAnchorRotation();

        glm::dvec3 y = observerPos;
        glm::dvec3 z = (observerRot * glm::dvec4(0, 0.1, -1, 0)).xyz();
        glm::dvec3 x = glm::cross(z, y);
        z            = glm::cross(x, y);

        x = glm::normalize(x);
        y = glm::normalize(y);
        z = glm::normalize(z);

        auto tilt     = glm::angleAxis(-0.2, glm::dvec3(1, 0, 0));
        auto rotation = glm::toQuat(glm::dmat3(x, y, z)) * tilt;

        mSolarSystem->flyObserverTo(mSolarSystem->getObserver().getCenterName(),
            mSolarSystem->getObserver().getFrameName(), observerPos, rotation,
            duration.value_or(3.0));
      }));

  // Flies the celestial observer to an orbit at three times the radius of the currently active
  // celestial body.
  mGuiManager->getGui()->registerCallback("navigation.toOrbit",
      "Increases the altitude of the observer significantly. The optional argument specifies the "
      "animation time in seconds (default is 3s).",
      std::function([this](std::optional<double> duration) {
        auto observerRot = mSolarSystem->getObserver().getAnchorRotation();
        auto radii = cs::core::SolarSystem::getRadii(mSolarSystem->getObserver().getCenterName());

        if (radii[0] == 0.0) {
          radii = glm::dvec3(1, 1, 1);
        }

        auto dir  = glm::normalize(mSolarSystem->getObserver().getAnchorPosition());
        auto cart = radii[0] * 3.0 * dir;

        glm::dvec3 y = (observerRot * glm::dvec4(0, -0.1, 1, 0)).xyz();
        glm::dvec3 z = dir;
        glm::dvec3 x = glm::cross(z, y);
        y            = glm::cross(z, x);

        x = glm::normalize(x);
        y = glm::normalize(y);
        z = glm::normalize(z);

        auto rotation = glm::toQuat(glm::dmat3(x, y, z));

        mSolarSystem->flyObserverTo(mSolarSystem->getObserver().getCenterName(),
            mSolarSystem->getObserver().getFrameName(), cart, rotation, duration.value_or(3.0));
      }));
}

////////////////////////////////////////////////////////////////////////////////////////////////////

void Application::unregisterGuiCallbacks() {
  mGuiManager->getGui()->unregisterCallback("core.listPlugins");
  mGuiManager->getGui()->unregisterCallback("core.loadPlugin");
  mGuiManager->getGui()->unregisterCallback("core.reloadPlugin");
  mGuiManager->getGui()->unregisterCallback("core.unloadPlugin");
  mGuiManager->getGui()->unregisterCallback("graphics.setAmbientLight");
  mGuiManager->getGui()->unregisterCallback("graphics.setEnableCascadesDebug");
  mGuiManager->getGui()->unregisterCallback("graphics.setEnableLighting");
  mGuiManager->getGui()->unregisterCallback("graphics.setEnableShadowFreeze");
  mGuiManager->getGui()->unregisterCallback("graphics.setEnableShadows");
  mGuiManager->getGui()->unregisterCallback("graphics.setEnableTimerQueries");
  mGuiManager->getGui()->unregisterCallback("graphics.setEnableVsync");
  mGuiManager->getGui()->unregisterCallback("graphics.setLightingQuality");
  mGuiManager->getGui()->unregisterCallback("graphics.setShadowmapBias");
  mGuiManager->getGui()->unregisterCallback("graphics.setShadowmapCascades");
  mGuiManager->getGui()->unregisterCallback("graphics.setShadowmapExtension");
  mGuiManager->getGui()->unregisterCallback("graphics.setShadowmapRange");
  mGuiManager->getGui()->unregisterCallback("graphics.setShadowmapResolution");
  mGuiManager->getGui()->unregisterCallback("graphics.setShadowmapSplitDistribution");
  mGuiManager->getGui()->unregisterCallback("graphics.setTerrainHeight");
  mGuiManager->getGui()->unregisterCallback("graphics.setWidgetScale");
  mGuiManager->getGui()->unregisterCallback("graphics.setFocalLength");
  mGuiManager->getGui()->unregisterCallback("graphics.setEnableAutoExposure");
  mGuiManager->getGui()->unregisterCallback("graphics.setEnableHDR");
  mGuiManager->getGui()->unregisterCallback("graphics.setExposure");
  mGuiManager->getGui()->unregisterCallback("graphics.setExposureAdaptionSpeed");
  mGuiManager->getGui()->unregisterCallback("graphics.setExposureCompensation");
  mGuiManager->getGui()->unregisterCallback("graphics.setSensorDiagonal");
  mGuiManager->getGui()->unregisterCallback("graphics.setEnableAutoGlow");
  mGuiManager->getGui()->unregisterCallback("graphics.setGlowIntensity");
  mGuiManager->getGui()->unregisterCallback("graphics.setExposureRange");
  mGuiManager->getGui()->unregisterCallback("navigation.fixHorizon");
  mGuiManager->getGui()->unregisterCallback("navigation.northUp");
  mGuiManager->getGui()->unregisterCallback("navigation.setBody");
  mGuiManager->getGui()->unregisterCallback("navigation.setBodyLongLatHeightDuration");
  mGuiManager->getGui()->unregisterCallback("navigation.setPosition");
  mGuiManager->getGui()->unregisterCallback("navigation.setRotation");
  mGuiManager->getGui()->unregisterCallback("navigation.toOrbit");
  mGuiManager->getGui()->unregisterCallback("navigation.toSurface");
  mGuiManager->getGui()->unregisterCallback("time.addHours");
  mGuiManager->getGui()->unregisterCallback("time.reset");
  mGuiManager->getGui()->unregisterCallback("time.set");
  mGuiManager->getGui()->unregisterCallback("time.setDate");
  mGuiManager->getGui()->unregisterCallback("time.setSpeed");
}
