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
#include "../cs-utils/utils.hpp"
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
#else
#define OPENLIB(libname) LoadLibrary((libname))
#define LIBFUNC(handle, fn) GetProcAddress((HMODULE)(handle), (fn))
#define CLOSELIB(handle) FreeLibrary((HMODULE)(handle))
#define LIBERROR() GetLastError()
#define LIBFILETYPE ".dll"
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
  pNodeFactory->SetNodeCreator("ObserverNavigationNode",
      new ObserverNavigationNodeCreate(mSolarSystem.get(), mInputManager.get()));

  // This connects several parts of CosmoScout VR to each other.
  connectSlots();

  // Setup user interface callbacks.
  registerGuiCallbacks();

  // add some hot-keys -----------------------------------------------------------------------------

  // '+' increases the speed of time.
  GetVistaSystem()->GetKeyboardSystemControl()->BindAction('+', [this]() {
    if (!mInputManager->pHoveredGuiNode.get()) {
      mTimeControl->increaseTimeSpeed();
    }
  });

  // '-' decreases the speed of time.
  GetVistaSystem()->GetKeyboardSystemControl()->BindAction('-', [this]() {
    if (!mInputManager->pHoveredGuiNode.get()) {
      mTimeControl->decreaseTimeSpeed();
    }
  });

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
    mRayTrans->SetScale(0.001, 0.001, 30);
    mRayTrans->SetName("Ray_Trans");

    auto ray      = new cs::graphics::MouseRay();
    auto coneNode = sceneGraph->NewOpenGLNode(mRayTrans, ray);
    coneNode->SetName("Cone_Node");

    VistaOpenSGMaterialTools::SetSortKeyOnSubtree(
        pIntentionNode, static_cast<int>(cs::utils::DrawOrder::eRay));
  } else {
    cs::core::GuiManager::setCursor(cs::gui::Cursor::ePointer);
  }

  mGuiManager->enableLoadingScreen(true);

  // open plugin libraries --------------------------------------------------------------------

  for (auto const& plugin : mSettings->mPlugins) {
    try {

#ifdef __linux__
      std::string path = "../share/plugins/lib" + plugin.first + ".so";
#else
      std::string path = "..\\share\\plugins\\" + plugin.first + ".dll";
#endif

      // Clear errors.
      LIBERROR();

      COSMOSCOUT_LIBTYPE pluginHandle = OPENLIB(path.c_str());

      if (pluginHandle) {
        cs::core::PluginBase* (*pluginConstructor)();
        pluginConstructor = (cs::core::PluginBase * (*)()) LIBFUNC(pluginHandle, "create");

        spdlog::info("Opening plugin '{}'.", plugin.first);

        // Actually call the plugin's constructor and add the returned pointer to out list.
        mPlugins.insert(
            std::pair<std::string, Plugin>(plugin.first, {pluginHandle, pluginConstructor()}));
      } else {
        spdlog::error("Failed to load plugin '{}': {}", plugin.first, LIBERROR());
      }
    } catch (std::exception const& e) {
      spdlog::error("Failed to load plugin '{}': {}", plugin.first, e.what());
    }
  }

  return VistaFrameLoop::Init(pVistaSystem);
}

////////////////////////////////////////////////////////////////////////////////////////////////////

void Application::Quit() {

  // De-init all plugins first.
  for (auto const& plugin : mPlugins) {
    plugin.second.mPlugin->deInit();
  }

  // Then close all plugins.
  for (auto const& plugin : mPlugins) {
    spdlog::info("Closing plugin '{}'.", plugin.first);

    auto handle           = plugin.second.mHandle;
    auto pluginDestructor = (void (*)(cs::core::PluginBase*))LIBFUNC(handle, "destroy");

    pluginDestructor(plugin.second.mPlugin);
    CLOSELIB(handle);
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
    mGuiManager->setLoadingScreenProgress(mDownloader->getProgress(), false);
  }

  // Once the data download has finished, we can delete our downloader.
  if (!mDownloadedData && mDownloader && mDownloader->hasFinished()) {
    mDownloadedData = true;
    mDownloader.release();
  }

  // If all data is available, we can initialize the SolarSystem. This can only be done after the
  // data download, as it requires SPICE kernels which might be part of the download.
  if (mDownloadedData && !mSolarSystem->getIsInitialized()) {
    mSolarSystem->init(mSettings->mSpiceKernel);

    // Store the frame at which we should start loading the plugins.
    mStartPluginLoadingAtFrame = GetFrameCount();
  }

  // load plugins ----------------------------------------------------------------------------------

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

      if (pluginToLoad >= 0 && pluginToLoad < mPlugins.size()) {

        // Get an iterator pointing to the plugin handle.
        auto plugin = mPlugins.begin();
        std::advance(plugin, pluginToLoad);

        // First provide the plugin with all required class instances.
        plugin->second.mPlugin->setAPI(mSettings, mSolarSystem, mGuiManager, mInputManager,
            GetVistaSystem()->GetGraphicsManager()->GetSceneGraph(), mGraphicsEngine, mFrameTimings,
            mTimeControl);

        // Then do the actual initialization. This may actually take a while and the loading screen
        // will become unresponsive in the meantime.
        try {
          plugin->second.mPlugin->init();
        } catch (std::exception const& e) {
          spdlog::error("Failed to initialize plugin '{}': {}", plugin->first, e.what());
        }

      } else if (pluginToLoad == mPlugins.size()) {

        spdlog::info("Ready for Takeoff!");

        // Once all plugins have been loaded, we set a boolean indicating this state.
        mLoadedAllPlugins = true;

        // Update the loading screen status.
        mGuiManager->setLoadingScreenStatus("Ready for Takeoff");
        mGuiManager->setLoadingScreenProgress(100.f, true);

        // All plugins finished loading -> init their custom components.
        mGuiManager->getGui()->callJavascript("CosmoScout.initInputs");

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
      if (pluginToLoad + 1 < mPlugins.size()) {
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
      mGuiManager->getGui()->callJavascript("CosmoScout.initInputs");
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

    // Update the navigation, SolarSystem and scene scale.
    {
      cs::utils::FrameTimings::ScopedTimer timer(
          "SolarSystem Update", cs::utils::FrameTimings::QueryMode::eCPU);
      mDragNavigation->update();
      mSolarSystem->update();
      mSolarSystem->updateSceneScale();
      mSolarSystem->updateObserverFrame();
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
      mGraphicsEngine->setSunDirection(glm::normalize(sunTransform[3].xyz()));
    }
  }

  // Update the user interface.
  {
    cs::utils::FrameTimings::ScopedTimer timer("User Interface");

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
        mGuiManager->getGui()->callJavascript("CosmoScout.statusbar.setObserverPosition",
            cs::utils::convert::toDegrees(polar.x), cs::utils::convert::toDegrees(polar.y),
            heightDiff);
      }

      // Update the compass in the header bar.
      auto rot = mSolarSystem->getObserver().getRelativeRotation(
          mTimeControl->pSimulationTime.get(), *mSolarSystem->pActiveBody.get());
      glm::dvec4 up(0.0, 1.0, 0.0, 0.0);
      glm::dvec4 north = rot * up;
      north.z          = 0.0;

      float angle = std::acos(glm::dot(up, glm::normalize(north)));
      if (north.x < 0.f) {
        angle = -angle;
      }

      mGuiManager->getGui()->callJavascript("CosmoScout.timeline.setNorthDirection", angle);
    }

    mGuiManager->update();
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
#ifdef __linux__
  std::string path = "../share/plugins";
#else
  std::string path = "..\\share\\plugins";
#endif

  auto plugins = cs::utils::filesystem::listFiles(path);

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

void Application::connectSlots() {

  // Update mouse pointer coordinate display in the user interface.
  mInputManager->pHoveredObject.onChange().connect(
      [this](cs::core::InputManager::Intersection intersection) {
        if (intersection.mObject) {
          auto body = std::dynamic_pointer_cast<cs::scene::CelestialBody>(intersection.mObject);

          if (body) {
            auto radii = body->getRadii();
            auto polar =
                cs::utils::convert::toLngLatHeight(intersection.mPosition, radii[0], radii[0]);
            auto lngLat = cs::utils::convert::toDegrees(polar.xy());

            if (!std::isnan(lngLat.x) && !std::isnan(lngLat.y) && !std::isnan(polar.z)) {
              mGuiManager->getGui()->callJavascript("CosmoScout.statusbar.setPointerPosition", true,
                  lngLat.x, lngLat.y, polar.z / mGraphicsEngine->pHeightScale.get());
              return;
            }
          }
        }
        mGuiManager->getGui()->callJavascript("CosmoScout.statusbar.setPointerPosition", false);
      });

  // Update the time shown in the user interface when the simulation time changes.
  mTimeControl->pSimulationTime.onChange().connect([this](double val) {
    std::stringstream sstr;
    auto              facet = new boost::posix_time::time_facet();
    facet->format("%d-%b-%Y %H:%M:%S.%f");
    sstr.imbue(std::locale(std::locale::classic(), facet));
    sstr << cs::utils::convert::toBoostTime(val);
    mGuiManager->getGui()->callJavascript("CosmoScout.timeline.setDate", sstr.str());
  });

  // Update the simulation time speed shown in the user interface.
  mTimeControl->pTimeSpeed.onChange().connect([this](float val) {
    mGuiManager->getGui()->callJavascript("CosmoScout.timeline.setTimeSpeed", val);
  });

  // Show notification when the center name of the celestial observer changes.
  mSolarSystem->pObserverCenter.onChange().connect([this](std::string const& center) {
    if (center == "Solar System Barycenter") {
      mGuiManager->showNotification("Leaving " + mSolarSystem->pActiveBody.get()->getCenterName(),
          "Now travelling in free space.", "star");
    } else {
      mGuiManager->showNotification(
          "Approaching " + mSolarSystem->pActiveBody.get()->getCenterName(),
          "Position is locked to " + mSolarSystem->pActiveBody.get()->getCenterName() + ".",
          "public");
    }
    mGuiManager->getGui()->callJavascript("CosmoScout.statusbar.setActivePlanetCenter", center);
  });

  // Show notification when the frame name of the celestial observer changes.
  mSolarSystem->pObserverFrame.onChange().connect([this](std::string const& frame) {
    if (frame == "J2000") {
      mGuiManager->showNotification(
          "Stop tracking " + mSolarSystem->pActiveBody.get()->getCenterName(),
          "Orbit is not synced anymore.", "vpn_lock");
    } else {
      mGuiManager->showNotification("Tracking " + mSolarSystem->pActiveBody.get()->getCenterName(),
          "Orbit in sync with " + mSolarSystem->pActiveBody.get()->getCenterName() + ".",
          "vpn_lock");
    }
    mGuiManager->getGui()->callJavascript("CosmoScout.statusbar.setActivePlanetFrame", frame);
  });

  // Show the current speed of the celestial observer in the user interface.
  mSolarSystem->pCurrentObserverSpeed.onChange().connect([this](float speed) {
    mGuiManager->getGui()->callJavascript("CosmoScout.statusbar.setObserverSpeed", speed);
  });

  // Show the statistics GuiItem when measurements are enabled.
  mFrameTimings->pEnableMeasurements.onChange().connect(
      [this](bool enable) { mGuiManager->getStatistics()->setIsEnabled(enable); });
}

////////////////////////////////////////////////////////////////////////////////////////////////////

void Application::registerGuiCallbacks() {

  // Shows a notification in the top right corner. See GuiManager::showNotification() for details.
  mGuiManager->getGui()->registerCallback<std::string, std::string, std::string>(
      "print_notification",
      ([this](std::string const& title, std::string const& content, std::string const& icon) {
        mGuiManager->showNotification(title, content, icon);
      }));

  // Flies the observer to the given celestial body.
  mGuiManager->getGui()->registerCallback<std::string>(
      "set_celestial_body", ([this](std::string const& name) {
        for (auto const& body : mSolarSystem->getBodies()) {
          if (body->getCenterName() == name) {
            mSolarSystem->pActiveBody = body;
            mSolarSystem->flyObserverTo(body->getCenterName(), body->getFrameName(), 10.0);
            mGuiManager->showNotification("Travelling", "to " + name, "send");
            break;
          }
        }
      }));

  // Sets the current simulation time. The argument must be a string accepted by
  // TimeControl::setTime.
  mGuiManager->getGui()->registerCallback<std::string>(
      "set_date", ([this](std::string const& sDate) {
        double time = cs::utils::convert::toSpiceTime(boost::posix_time::time_from_string(sDate));
        mTimeControl->setTime(time);
      }));

  // Sets the current simulation time. The argument must be a double representing Barycentric
  // Dynamical Time.
  mGuiManager->getGui()->registerCallback<double>(
      "set_time", ([this](double tTime) { mTimeControl->setTime(tTime); }));

  // Adjusts the global ambient brightness factor.
  mGuiManager->getGui()->registerCallback<double>(
      "set_ambient_light", ([this](double value) { mGraphicsEngine->pAmbientBrightness = value; }));

  // Enables lighting computation globally.
  mGuiManager->getGui()->registerCallback<bool>(
      "set_enable_lighting", ([this](bool enable) { mGraphicsEngine->pEnableLighting = enable; }));

  // Shows cascaded shadow mapping debugging information on the terrain.
  mGuiManager->getGui()->registerCallback<bool>("set_enable_cascades_debug",
      ([this](bool enable) { mGraphicsEngine->pEnableShadowsDebug = enable; }));

  // Enables the calculation of shadows.
  mGuiManager->getGui()->registerCallback<bool>(
      "set_enable_shadows", ([this](bool enable) { mGraphicsEngine->pEnableShadows = enable; }));

  // Freezes the shadow frustum.
  mGuiManager->getGui()->registerCallback<bool>("set_enable_shadow_freeze",
      ([this](bool enable) { mGraphicsEngine->pEnableShadowsFreeze = enable; }));

  // Sets a value which individual plugins may honor trading rendering fidelity for performance.
  mGuiManager->getGui()->registerCallback<double>("set_lighting_quality",
      ([this](const int value) { mGraphicsEngine->pLightingQuality = value; }));

  // Adjusts the resolution of the shadowmap.
  mGuiManager->getGui()->registerCallback<double>("set_shadowmap_resolution",
      ([this](const int val) { mGraphicsEngine->pShadowMapResolution = val; }));

  // Adjusts the number of shadowmap cascades.
  mGuiManager->getGui()->registerCallback<double>("set_shadowmap_cascades",
      ([this](const int val) { mGraphicsEngine->pShadowMapCascades = val; }));

  // Adjusts the depth range of the shadowmap.
  mGuiManager->getGui()->registerCallback<double, double>(
      "set_shadowmap_range", ([this](double val, double handle) {
        glm::vec2 range = mGraphicsEngine->pShadowMapRange.get();

        if (handle == 0.0) {
          range.x = (float)val;
        } else {
          range.y = (float)val;
        };

        mGraphicsEngine->pShadowMapRange = range;
      }));

  // Adjusts the additional frustum length for shadowmap rendering in sun space.
  mGuiManager->getGui()->registerCallback<double, double>(
      "set_shadowmap_extension", ([this](double val, double handle) {
        glm::vec2 extension = mGraphicsEngine->pShadowMapExtension.get();

        if (handle == 0.0) {
          extension.x = (float)val;
        } else {
          extension.y = (float)val;
        };

        mGraphicsEngine->pShadowMapExtension = extension;
      }));

  // Adjusts the distribution of shadowmap cascades.
  mGuiManager->getGui()->registerCallback<double>("set_shadowmap_split_distribution",
      ([this](double val) { mGraphicsEngine->pShadowMapSplitDistribution = val; }));

  // Adjusts the bias to mitigate shadow acne.
  mGuiManager->getGui()->registerCallback<double>(
      "set_shadowmap_bias", ([this](double val) { mGraphicsEngine->pShadowMapBias = val; }));

  // A global factor which plugins may honor when they render some sort of terrain.
  mGuiManager->getGui()->registerCallback<double>(
      "set_terrain_height", ([this](double value) { mGraphicsEngine->pHeightScale = value; }));

  // Adjusts the global scaling of world-space widgets.
  mGuiManager->getGui()->registerCallback<double>(
      "set_widget_scale", ([this](double value) { mGraphicsEngine->pWidgetScale = value; }));

  // Enables or disables the per-frame time measurements.
  mGuiManager->getGui()->registerCallback<bool>("set_enable_timer_queries",
      ([this](bool value) { mFrameTimings->pEnableMeasurements = value; }));

  // Enables or disables vertical synchronization.
  mGuiManager->getGui()->registerCallback<bool>("set_enable_vsync", ([this](bool value) {
    GetVistaSystem()
        ->GetDisplayManager()
        ->GetWindows()
        .begin()
        ->second->GetWindowProperties()
        ->SetVSyncEnabled(value);
  }));

  // Timeline callbacks ----------------------------------------------------------------------------

  mGuiManager->getGui()->registerCallback("reset_time", ([this]() { mTimeControl->resetTime(); }));

  mGuiManager->getGui()->registerCallback<double>("add_hours", ([this](double amount) {
    mTimeControl->setTime(mTimeControl->pSimulationTime.get() + 60.0 * 60.0 * amount);
  }));

  mGuiManager->getGui()->registerCallback<double>(
      "add_hours_without_animation", ([this](double amount) {
        mTimeControl->setTimeWithoutAnimation(
            mTimeControl->pSimulationTime.get() + 60.0 * 60.0 * amount);
      }));

  mGuiManager->getGui()->registerCallback<double>(
      "set_time_speed", ([this](double speed) { mTimeControl->setTimeSpeed(speed); }));

  // Flies the celestial observer to the given location in space.
  mGuiManager->getGui()->registerCallback<std::string, double, double, double, double>(
      "fly_to_location", ([this](std::string const& name, double longitude, double latitude,
                              double height, double time) {
        for (auto const& body : mSolarSystem->getBodies()) {
          if (body->getCenterName() == name) {
            mSolarSystem->pActiveBody = body;
            mSolarSystem->flyObserverTo(body->getCenterName(), body->getFrameName(),
                cs::utils::convert::toRadians(glm::dvec2(longitude, latitude)), height, time);
          }
        }
      }));

  // Rotates the scene in such a way, that the y-axis points towards the north pole of the currently
  // active celestial body.
  mGuiManager->getGui()->registerCallback("navigate_north_up", [this]() {
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
        mSolarSystem->getObserver().getFrameName(), observerPos, rotation, 1.0);
  });

  // Rotates the scene in such a way, that the currently visible horizon is levelled.
  mGuiManager->getGui()->registerCallback("navigate_fix_horizon", [this]() {
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
        mSolarSystem->getObserver().getFrameName(), observerPos, rotation, 1.0);
  });

  // Flies the celestial observer to 0.1% of its current height.
  mGuiManager->getGui()->registerCallback("navigate_to_surface", [this]() {
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
        mSolarSystem->getObserver().getFrameName(), observerPos, rotation, 3.0);
  });

  // Flies the celestial observer to an orbit at three times the radius of the currently active
  // celestial body.
  mGuiManager->getGui()->registerCallback("navigate_to_orbit", [this]() {
    auto observerRot = mSolarSystem->getObserver().getAnchorRotation();
    auto radii       = cs::core::SolarSystem::getRadii(mSolarSystem->getObserver().getCenterName());

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
        mSolarSystem->getObserver().getFrameName(), cart, rotation, 3.0);
  });
}

////////////////////////////////////////////////////////////////////////////////////////////////////

void Application::unregisterGuiCallbacks() {
  mGuiManager->getGui()->unregisterCallback("set_lighting_quality");
  mGuiManager->getGui()->unregisterCallback("set_celestial_body");
  mGuiManager->getGui()->unregisterCallback("set_enable_shadows");
  mGuiManager->getGui()->unregisterCallback("print_notification");
  mGuiManager->getGui()->unregisterCallback("set_shadowmap_split_distribution");
  mGuiManager->getGui()->unregisterCallback("set_enable_cascades_debug");
  mGuiManager->getGui()->unregisterCallback("set_ambient_light");
  mGuiManager->getGui()->unregisterCallback("set_shadowmap_cascades");
  mGuiManager->getGui()->unregisterCallback("set_shadowmap_extension");
  mGuiManager->getGui()->unregisterCallback("set_enable_shadow_freeze");
  mGuiManager->getGui()->unregisterCallback("set_date");
  mGuiManager->getGui()->unregisterCallback("set_time");
  mGuiManager->getGui()->unregisterCallback("set_shadowmap_range");
  mGuiManager->getGui()->unregisterCallback("set_terrain_height");
  mGuiManager->getGui()->unregisterCallback("set_widget_scale");
  mGuiManager->getGui()->unregisterCallback("set_enable_lighting");
  mGuiManager->getGui()->unregisterCallback("set_enable_timer_queries");
  mGuiManager->getGui()->unregisterCallback("set_shadowmap_resolution");
  mGuiManager->getGui()->unregisterCallback("set_shadowmap_bias");
  mGuiManager->getGui()->unregisterCallback("set_enable_vsync");
  mGuiManager->getGui()->unregisterCallback("navigate_to_surface");
  mGuiManager->getGui()->unregisterCallback("fly_to_location");
  mGuiManager->getGui()->unregisterCallback("reset_time");
  mGuiManager->getGui()->unregisterCallback("navigate_to_orbit");
  mGuiManager->getGui()->unregisterCallback("set_time_speed");
  mGuiManager->getGui()->unregisterCallback("add_hours_without_animation");
  mGuiManager->getGui()->unregisterCallback("navigate_fix_horizon");
  mGuiManager->getGui()->unregisterCallback("navigate_north_up");
  mGuiManager->getGui()->unregisterCallback("add_hours");
}
