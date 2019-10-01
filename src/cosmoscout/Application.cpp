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
#include "../cs-utils/utils.hpp"
#include "ObserverNavigationNode.hpp"

#include <curlpp/cURLpp.hpp>
#include <glm/gtx/quaternion.hpp>

#include <VistaBase/VistaTimeUtils.h>
#include <VistaInterProcComm/Cluster/VistaClusterDataSync.h>
#include <VistaKernel/Cluster/VistaClusterMode.h>
#include <VistaKernel/DisplayManager/GlutWindowImp/VistaGlutWindowingToolkit.h>
#include <VistaKernel/DisplayManager/VistaDisplayManager.h>
#include <VistaKernel/DisplayManager/VistaProjection.h>
#include <VistaKernel/EventManager/VistaEventManager.h>
#include <VistaKernel/EventManager/VistaSystemEvent.h>
#include <VistaKernel/GraphicsManager/VistaOpenGLNode.h>
#include <VistaKernel/GraphicsManager/VistaSceneGraph.h>
#include <VistaKernel/GraphicsManager/VistaTransformNode.h>
#include <VistaKernel/VistaSystem.h>
#include <VistaKernelOpenSGExt/VistaOpenSGMaterialTools.h>

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
}

////////////////////////////////////////////////////////////////////////////////////////////////////

Application::~Application() {
  for (auto const& plugin : mPlugins) {
    plugin.second.mPlugin->deInit();
  }

  for (auto const& plugin : mPlugins) {
    std::string pluginFile = plugin.first;
    std::cout << "Unloading Plugin " << pluginFile << std::endl;

    auto handle           = plugin.second.mHandle;
    auto pluginDestructor = (void (*)(cs::core::PluginBase*))LIBFUNC(handle, "destroy");

    pluginDestructor(plugin.second.mPlugin);
    CLOSELIB(handle);
  }

  mPlugins.clear();

  mSolarSystem->deinit();

  cURLpp::terminate();
}

////////////////////////////////////////////////////////////////////////////////////////////////////

bool Application::Init(VistaSystem* pVistaSystem) {
  // initialize curl
  cURLpp::initialize();

  mInputManager   = std::make_shared<cs::core::InputManager>();
  mFrameTimings   = std::make_shared<cs::utils::FrameTimings>();
  mGraphicsEngine = std::make_shared<cs::core::GraphicsEngine>(mSettings);
  mGuiManager     = std::make_shared<cs::core::GuiManager>(mSettings, mInputManager, mFrameTimings);
  mSceneSync =
      std::unique_ptr<IVistaClusterDataSync>(GetVistaSystem()->GetClusterMode()->CreateDataSync());
  mTimeControl = std::make_shared<cs::core::TimeControl>(mSettings);
  mSolarSystem = std::make_shared<cs::core::SolarSystem>(mTimeControl);
  mDragNavigation =
      std::make_shared<cs::core::DragNavigation>(mSolarSystem, mInputManager, mTimeControl);

  connectSlots();

  // clang-format off
  VdfnNodeFactory* pNodeFactory = VdfnNodeFactory::GetSingleton();
  pNodeFactory->SetNodeCreator("ObserverNavigationNode", new ObserverNavigationNodeCreate(mSolarSystem, mInputManager));
  // clang-format on

  // add hot-keys ----------------------------------------------------------------------------------
  GetVistaSystem()->GetKeyboardSystemControl()->BindAction('+', [this]() {
    if (!mInputManager->pHoveredGuiNode.get()) {
      mTimeControl->increaseTimeSpeed();
    }
  });

  GetVistaSystem()->GetKeyboardSystemControl()->BindAction('-', [this]() {
    if (!mInputManager->pHoveredGuiNode.get()) {
      mTimeControl->decreaseTimeSpeed();
    }
  });

  // set mouse pointer -----------------------------------------------------------------------------

  auto windowingToolkit = dynamic_cast<VistaGlutWindowingToolkit*>(
      GetVistaSystem()->GetDisplayManager()->GetWindowingToolkit());

  if (windowingToolkit) {
    for (auto const& window : GetVistaSystem()->GetDisplayManager()->GetWindows()) {
      windowingToolkit->SetCursorIsEnabled(window.second, !mSettings->mEnableMouseRay);
    }
  }

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
    mGuiManager->setCursor(cs::gui::Cursor::ePointer);
  }

  // open plugins ----------------------------------------------------------------------------------

  for (auto const& plugin : mSettings->mPlugins) {
    try {

#ifdef __linux__
      std::string path = "../share/plugins/lib" + plugin.first + ".so";
#else
      std::string path = "..\\share\\plugins\\" + plugin.first + ".dll";
#endif

      // Clear errors
      LIBERROR();

      COSMOSCOUT_LIBTYPE pluginHandle = OPENLIB(path.c_str());

      if (pluginHandle) {
        cs::core::PluginBase* (*pluginConstructor)();
        pluginConstructor = (cs::core::PluginBase * (*)()) LIBFUNC(pluginHandle, "create");

        std::cout << "Opening Plugin " << plugin.first << " ..." << std::endl;

        mPlugins.insert(
            std::pair<std::string, Plugin>(plugin.first, {pluginHandle, pluginConstructor()}));

      } else {
        std::cerr << "Error loading CosmoScout VR Plugin " << plugin.first << " : " << LIBERROR()
                  << std::endl;
      }
    } catch (std::exception const& e) {
      std::cerr << "Error loading plugin " << plugin.first << ": " << e.what() << std::endl;
    }
  }

  return VistaFrameLoop::Init(pVistaSystem);
}

////////////////////////////////////////////////////////////////////////////////////////////////////

void Application::FrameUpdate() {
  ++m_iFrameCount;

  mFrameTimings->startFullFrameTiming();

  // emit vista events
  {
    cs::utils::FrameTimings::ScopedTimer timer("ClusterMode StartFrame");
    m_pClusterMode->StartFrame();
  }

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

  const int32_t startLoadingAtFrame = 150;

  if (GetFrameCount() == startLoadingAtFrame) {
    // download datasets if required
    if (mSettings->mDownloadData.size() > 0) {
      mDownloader.reset(new cs::utils::Downloader(10));
      for (auto const& download : mSettings->mDownloadData) {
        mDownloader->download(download.mUrl, download.mFile);
      }
    } else {
      mDownloadedData = true;
    }
  }

  if (!mDownloadedData && mDownloader && mDownloader->hasFinished()) {
    mDownloadedData = true;
    mDownloader.release();
  }

  if (mDownloadedData && !mSolarSystem->getIsInitialized()) {
    mSolarSystem->init(mSettings->mSpiceKernel);
    mStartPluginLoadingAtFrame = GetFrameCount();
  }

  // load plugins ----------------------------------------------------------------------------------
  // Start loading of resources after a short delay to make sure that the loading screen is visible.
  // Then we will draw some frames between each plugin to be able to update the loading screen's
  // status message.
  if (mDownloadedData && !mLoadedAllPlugins) {
    const int32_t loadingDelayFrames = 25;

    int32_t nextPluginToLoad = (GetFrameCount() - mStartPluginLoadingAtFrame) / loadingDelayFrames;

    if (nextPluginToLoad >= 0 && nextPluginToLoad < mPlugins.size()) {
      auto plugin = mPlugins.begin();
      std::advance(plugin, nextPluginToLoad);
      mGuiManager->setLoadingScreenStatus("Loading " + plugin->first + " ...");
    }

    if ((std::max(0, GetFrameCount() - mStartPluginLoadingAtFrame) % loadingDelayFrames) == 0) {
      int32_t pluginToLoad = nextPluginToLoad - 1;
      if (pluginToLoad >= 0 && pluginToLoad < mPlugins.size()) {

        // load plugin -----------------------------------------------------------------------------
        auto plugin = mPlugins.begin();
        std::advance(plugin, pluginToLoad);

        auto sceneGraph = GetVistaSystem()->GetGraphicsManager()->GetSceneGraph();

        plugin->second.mPlugin->setAPI(mSettings, mSolarSystem, mGuiManager, mInputManager,
            sceneGraph, mGraphicsEngine, mFrameTimings, mTimeControl);

        try {
          plugin->second.mPlugin->init();
        } catch (std::exception const& e) {
          std::cerr << "Error initializing plugin " << plugin->first << ": " << e.what()
                    << std::endl;
        }

      } else if (pluginToLoad == mPlugins.size()) {

        mLoadedAllPlugins = true;

        // initial observer animation
        auto const& observerSettings = mSettings->mObserver;
        glm::dvec2  lonLat(observerSettings.mLongitude, observerSettings.mLatitude);
        lonLat = cs::utils::convert::toRadians(lonLat);

        auto radii = cs::core::SolarSystem::getRadii(observerSettings.mCenter);

        if (radii[0] == 0.0 || radii[2] == 0.0) {
          radii = glm::dvec3(1, 1, 1);
        }

        // multiply longitude and latitude of start location by 0.5 to create
        // more interesting start animation
        auto cart = cs::utils::convert::toCartesian(
            lonLat * 0.5, radii[0], radii[0], observerSettings.mDistance * 10);

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

        mGuiManager->hideLoadingScreen();
        std::cout << "Loading done." << std::endl;
      }
    }
  }

  if (mLoadedAllPlugins) {
    // update CosmoScout VR classes
    {
      cs::utils::FrameTimings::ScopedTimer timer(
          "TimeControl Update", cs::utils::FrameTimings::QueryMode::eCPU);
      mTimeControl->update();
    }

    for (auto const& plugin : mPlugins) {
      cs::utils::FrameTimings::ScopedTimer timer(
          plugin.first, cs::utils::FrameTimings::QueryMode::eBoth);

      try {
        plugin.second.mPlugin->update();
      } catch (std::runtime_error const& e) {
        std::cerr << "Error updating plugin " << plugin.first << ": " << e.what() << std::endl;
      }
    }

    {
      cs::utils::FrameTimings::ScopedTimer timer(
          "SolarSystem Update", cs::utils::FrameTimings::QueryMode::eCPU);
      mDragNavigation->update();
      mSolarSystem->update();
      updateSceneScale();
    }

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

    {
      auto sunTransform = mSolarSystem->getSun()->getWorldTransform();
      mGraphicsEngine->setSunDirection(glm::normalize(sunTransform[3].xyz()));
    }

    if (mSolarSystem->pActiveBody.get()) {
      glm::dmat4 mPTransInv = glm::inverse(mSolarSystem->pActiveBody.get()->getWorldTransform());

      auto                pSG = GetVistaSystem()->GetGraphicsManager()->GetSceneGraph();
      VistaTransformNode* pTrans =
          dynamic_cast<VistaTransformNode*>(pSG->GetNode("Platform-User-Node"));

      auto vWorldPos = glm::vec4(1);
      pTrans->GetWorldPosition(vWorldPos.x, vWorldPos.y, vWorldPos.z);

      // transform user position into planet coordinate system
      auto   radii      = mSolarSystem->pActiveBody.get()->getRadii();
      auto   vPlanetPos = mPTransInv * vWorldPos;
      auto   polar      = cs::utils::convert::toLngLatHeight(vPlanetPos.xyz(), radii[0], radii[0]);
      double surfaceHeight = mSolarSystem->pActiveBody.get()->getHeight(polar.xy());
      double heightDiff    = polar.z / mGraphicsEngine->pHeightScale.get() - surfaceHeight;

      if (!std::isnan(polar.x) && !std::isnan(polar.y) && !std::isnan(heightDiff)) {
        mGuiManager->getHeaderBar()->callJavascript("set_user_position",
            cs::utils::convert::toDegrees(polar.x), cs::utils::convert::toDegrees(polar.y),
            heightDiff);
      }

      // set compass ---------------------------------------------------------
      auto rot = mSolarSystem->getObserver().getRelativeRotation(
          mTimeControl->pSimulationTime.get(), *mSolarSystem->pActiveBody.get());
      glm::dvec4 up(0.0, 1.0, 0.0, 0.0);
      glm::dvec4 north = rot * up;
      north.z          = 0.0;

      float angle = std::acos(glm::dot(up, glm::normalize(north)));
      if (north.x < 0.f) {
        angle = -angle;
      }

      mGuiManager->getHeaderBar()->callJavascript("set_north_direction", angle);
    }
  }

  {
    cs::utils::FrameTimings::ScopedTimer timer(
        "User Interface", cs::utils::FrameTimings::QueryMode::eCPU);
    mGuiManager->update();
  }

  // update vista classes
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

  mFrameTimings->endFullFrameTiming();

  {
    cs::utils::FrameTimings::ScopedTimer timer("DisplayManager DisplayFrame");
    m_pDisplayManager->DisplayFrame();
  }

  {
    cs::utils::FrameTimings::ScopedTimer timer("FrameRate RecordTime");
    m_pFrameRate->RecordTime();
  }

  // record frame timings
  mFrameTimings->update();
}

////////////////////////////////////////////////////////////////////////////////////////////////////

void Application::connectSlots() {

  // UI updates based on property changes ----------------------------------------------------------
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
              mGuiManager->getHeaderBar()->callJavascript("set_pointer_position", true, lngLat.x,
                  lngLat.y, polar.z / mGraphicsEngine->pHeightScale.get());
              return;
            }
          }
        }
        mGuiManager->getHeaderBar()->callJavascript("set_pointer_position", false);
      });

  mTimeControl->pSimulationTime.onChange().connect([this](double val) {
    std::stringstream sstr;
    auto              facet = new boost::posix_time::time_facet();
    facet->format("%d-%b-%Y %H:%M");
    sstr.imbue(std::locale(std::locale::classic(), facet));
    sstr << cs::utils::convert::toBoostTime(val);
    mGuiManager->getHeaderBar()->callJavascript("set_date", sstr.str());
  });

  mTimeControl->pTimeSpeed.onChange().connect(
      [this](float val) { mGuiManager->getHeaderBar()->callJavascript("set_time_speed", val); });

  // register some callbacks -----------------------------------------------------------------------

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
  });

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
  });

  mSolarSystem->pCurrentObserverSpeed.onChange().connect(
      [this](float speed) { mGuiManager->getHeaderBar()->callJavascript("set_speed", speed); });

  mGuiManager->getSideBar()->registerCallback<std::string>(
      "set_celestial_body", ([this](std::string const& name) {
        for (auto const& body : mSolarSystem->getBodies()) {
          if (body->getCenterName() == name) {
            mSolarSystem->pActiveBody = body;
            mSolarSystem->flyObserverTo(body->getCenterName(), body->getFrameName(), 10.0);
          }
        }
      }));

  mGuiManager->getSideBar()->registerCallback<std::string>(
      "set_date", ([this](std::string const& sDate) {
        double time = cs::utils::convert::toSpiceTime(boost::posix_time::time_from_string(sDate));
        mTimeControl->setTime(time);
      }));

  mGuiManager->getSideBar()->registerCallback<std::string>(
      "set_date", ([this](std::string const& sDate) {
        double time = cs::utils::convert::toSpiceTime(boost::posix_time::time_from_string(sDate));
        mTimeControl->setTime(time);
      }));

  mGuiManager->getSideBar()->registerCallback<double>(
      "set_time", ([this](double tTime) { mTimeControl->setTime(tTime); }));

  mGuiManager->getHeaderBar()->registerCallback<std::string, std::string, double, double, double>(
      "fly_to", [this](std::string const& center, std::string const& frame, double longitude,
                    double latitude, double height) {
        mSolarSystem->flyObserverTo(center, frame,
            cs::utils::convert::toRadians(glm::dvec2(longitude, latitude)), height, 5.0);
      });

  mGuiManager->getHeaderBar()->registerCallback("navigate_north_up", [this]() {
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

  mGuiManager->getHeaderBar()->registerCallback("navigate_fix_horizon", [this]() {
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

  mGuiManager->getHeaderBar()->registerCallback("navigate_to_surface", [this]() {
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

  mGuiManager->getHeaderBar()->registerCallback("navigate_to_orbit", [this]() {
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

  mGuiManager->getHeaderBar()->registerCallback(
      "reset_time", ([this]() { mTimeControl->resetTime(); }));

  mGuiManager->getHeaderBar()->registerCallback("toggle_time_stop", ([&]() {
    float speed(mTimeControl->pTimeSpeed.get() == 0.f ? 1.f : 0.f);
    mTimeControl->pTimeSpeed = speed;
  }));

  mGuiManager->getHeaderBar()->registerCallback<double>("add_hours", ([&](double amount) {
    mTimeControl->setTime(mTimeControl->pSimulationTime.get() + 60.0 * 60.0 * amount);
  }));

  mGuiManager->getHeaderBar()->registerCallback(
      "increase_time_speed", ([&]() { mTimeControl->increaseTimeSpeed(); }));

  mGuiManager->getHeaderBar()->registerCallback(
      "decrease_time_speed", ([&]() { mTimeControl->decreaseTimeSpeed(); }));

  mGuiManager->getSideBar()->registerCallback<bool>(
      "set_enable_shadows", ([this](bool enable) { mGraphicsEngine->pEnableShadows = enable; }));

  mGuiManager->getSideBar()->registerCallback<bool>("set_enable_shadow_freeze",
      ([this](bool enable) { mGraphicsEngine->pEnableShadowsFreeze = enable; }));

  mGuiManager->getSideBar()->registerCallback<bool>("set_enable_cascades_debug",
      ([this](bool enable) { mGraphicsEngine->pEnableShadowsDebug = enable; }));

  mGuiManager->getSideBar()->registerCallback<bool>(
      "set_enable_lighting", ([this](bool enable) { mGraphicsEngine->pEnableLighting = enable; }));

  mGuiManager->getSideBar()->registerCallback<double>("set_lighting_quality",
      ([this](const int value) { mGraphicsEngine->pLightingQuality = value; }));

  mGuiManager->getSideBar()->registerCallback<double>(
      "set_ambient_light", ([this](double value) { mGraphicsEngine->pAmbientBrightness = value; }));

  mGuiManager->getSideBar()->registerCallback<double>("set_shadowmap_resolution",
      ([this](const int val) { mGraphicsEngine->pShadowMapResolution = val; }));

  mGuiManager->getSideBar()->registerCallback<double>("set_shadowmap_cascades",
      ([this](const int val) { mGraphicsEngine->pShadowMapCascades = val; }));

  mGuiManager->getSideBar()->registerCallback<double, double>(
      "set_shadowmap_range", ([this](double val, double handle) {
        glm::vec2 range = mGraphicsEngine->pShadowMapRange.get();

        if (handle == 0.0) {
          range.x = (float)val;
        } else {
          range.y = (float)val;
        };

        mGraphicsEngine->pShadowMapRange = range;
      }));

  mGuiManager->getSideBar()->registerCallback<double, double>(
      "set_shadowmap_extension", ([this](double val, double handle) {
        glm::vec2 extension = mGraphicsEngine->pShadowMapExtension.get();

        if (handle == 0.0) {
          extension.x = (float)val;
        } else {
          extension.y = (float)val;
        };

        mGraphicsEngine->pShadowMapExtension = extension;
      }));

  mGuiManager->getSideBar()->registerCallback<double>("set_shadowmap_split_distribution",
      ([this](double val) { mGraphicsEngine->pShadowMapSplitDistribution = val; }));

  mGuiManager->getSideBar()->registerCallback<double>(
      "set_shadowmap_bias", ([this](double val) { mGraphicsEngine->pShadowMapBias = val; }));

  mGuiManager->getSideBar()->registerCallback<double>(
      "set_terrain_height", ([this](double value) { mGraphicsEngine->pHeightScale = value; }));

  mGuiManager->getSideBar()->registerCallback<double>(
      "set_widget_scale", ([this](double value) { mGraphicsEngine->pWidgetScale = value; }));

  mGuiManager->getCalendar()->registerCallback<std::string>(
      "set_date", ([this](std::string const& date) {
        mTimeControl->setTime(
            cs::utils::convert::toSpiceTime(boost::posix_time::time_from_string(date)));
      }));
}

////////////////////////////////////////////////////////////////////////////////////////////////////

void Application::updateSceneScale() {
  auto&  oObs           = mSolarSystem->getObserver();
  double simulationTime = mTimeControl->pSimulationTime.get();

  // user will be locked to active planet, scene will be scaled that closest planet
  // is mScaleDistance away in world space
  std::shared_ptr<cs::scene::CelestialBody> pClosestBody;
  std::shared_ptr<cs::scene::CelestialBody> pActiveBody;

  double dActiveWeight    = 0;
  double dClosestDistance = std::numeric_limits<double>::max();

  glm::dvec3 vClosestPlanetObserverPosition(0.0);

  for (auto const& object : mSolarSystem->getBodies()) {
    if (!object->getIsInExistence()) {
      continue;
    }

    auto radii = object->getRadii();

    if (radii.x <= 0.0 || radii.y <= 0.0 || radii.z <= 0.0) {
      continue;
    }

    glm::dvec3 vObserverPos;

    try {
      vObserverPos = object->getRelativePosition(simulationTime, oObs);
    } catch (...) { continue; }

    double dDistance = glm::length(vObserverPos) - radii[0];
    double dWeight   = (radii[0] + mSettings->mSceneScale.mMinObjectSize) /
                     std::max(radii[0] + mSettings->mSceneScale.mMinObjectSize,
                         radii[0] + dDistance - mSettings->mSceneScale.mMinObjectSize);

    if (dWeight > dActiveWeight) {
      pActiveBody   = object;
      dActiveWeight = dWeight;
    }

    if (dDistance < dClosestDistance) {
      pClosestBody                   = object;
      dClosestDistance               = dDistance;
      vClosestPlanetObserverPosition = vObserverPos;
    }
  }

  // change frame and center if there is a object with weight larger than mLockWeight
  // and mTrackWeight
  if (pActiveBody) {
    if (!oObs.isAnimationInProgress()) {
      std::string sCenter = "Solar System Barycenter";
      std::string sFrame  = "J2000";

      if (dActiveWeight > mSettings->mSceneScale.mLockWeight) {
        sFrame = pActiveBody->getFrameName();
      }

      if (dActiveWeight > mSettings->mSceneScale.mTrackWeight) {
        sCenter = pActiveBody->getCenterName();
      }

      mSolarSystem->pActiveBody     = pActiveBody;
      mSolarSystem->pObserverCenter = sCenter;
      mSolarSystem->pObserverFrame  = sFrame;
    }
  }

  // scale scene in such a way that the closest planet
  // is mScaleDistance away in world space
  if (pClosestBody) {
    auto   dSurfaceHeight = 0.0;
    double dRealDistance  = glm::length(vClosestPlanetObserverPosition);

    auto radii = pClosestBody->getRadii();

    if (radii[0] > 0) {
      auto lngLatHeight =
          cs::utils::convert::toLngLatHeight(vClosestPlanetObserverPosition, radii[0], radii[0]);
      dRealDistance = lngLatHeight.z;
      dRealDistance -=
          pClosestBody->getHeight(lngLatHeight.xy()) * mGraphicsEngine->pHeightScale.get();
    }

    if (std::isnan(dRealDistance)) {
      return;
    }

    double interpolate = 1.0;

    if (mSettings->mSceneScale.mFarRealDistance != mSettings->mSceneScale.mCloseRealDistance) {
      interpolate = glm::clamp(
          (dRealDistance - mSettings->mSceneScale.mCloseRealDistance) /
              (mSettings->mSceneScale.mFarRealDistance - mSettings->mSceneScale.mCloseRealDistance),
          0.0, 1.0);
    }

    double dScale = dRealDistance / glm::mix(mSettings->mSceneScale.mCloseVisualDistance,
                                        mSettings->mSceneScale.mFarVisualDistance, interpolate);
    dScale = glm::clamp(dScale, mSettings->mSceneScale.mMinScale, mSettings->mSceneScale.mMaxScale);
    oObs.setAnchorScale(dScale);

    if (dRealDistance < mSettings->mSceneScale.mCloseRealDistance) {
      double     penetration = mSettings->mSceneScale.mCloseRealDistance - dRealDistance;
      glm::dvec3 position    = oObs.getAnchorPosition();
      oObs.setAnchorPosition(position + glm::normalize(position) * penetration);
    }

    // set far clip dynamically
    auto projections = GetVistaSystem()->GetDisplayManager()->GetProjections();
    for (auto const& projection : projections) {
      projection.second->GetProjectionProperties()->SetClippingRange(
          mSettings->mSceneScale.mNearClip, glm::mix(mSettings->mSceneScale.mMaxFarClip,
                                                mSettings->mSceneScale.mMinFarClip, interpolate));
    }
  }

  // update speed display
  static auto sLastObserverPosition = oObs.getAnchorPosition();

  mSolarSystem->pCurrentObserverSpeed =
      glm::length(sLastObserverPosition - oObs.getAnchorPosition()) /
      mFrameTimings->pFrameTime.get();
  sLastObserverPosition = oObs.getAnchorPosition();
}

////////////////////////////////////////////////////////////////////////////////////////////////////
