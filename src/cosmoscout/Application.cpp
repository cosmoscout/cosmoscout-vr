////////////////////////////////////////////////////////////////////////////////////////////////////
//                               This file is part of CosmoScout VR                               //
//      and may be used under the terms of the MIT license. See the LICENSE file for details.     //
//                        Copyright: (c) 2019 German Aerospace Center (DLR)                       //
////////////////////////////////////////////////////////////////////////////////////////////////////

#include "Application.hpp"

#include "../cs-core/GraphicsEngine.hpp"
#include "../cs-core/GuiManager.hpp"
#include "../cs-core/InputManager.hpp"
#include "../cs-core/SolarSystem.hpp"
#include "../cs-core/TimeControl.hpp"
#include "../cs-utils/convert.hpp"
#include "../cs-utils/utils.hpp"
#include "dfn-nodes/AutoSceneScaleNode.hpp"
#include "dfn-nodes/DragNavigationNode.hpp"
#include "dfn-nodes/ObserverNavigationNode.hpp"
#include "dfn-nodes/ObserverSyncNode.hpp"

#include <curlpp/cURLpp.hpp>

#include <VistaBase/VistaTimeUtils.h>
#include <VistaKernel/Cluster/VistaClusterMode.h>
#include <VistaKernel/DisplayManager/GlutWindowImp/VistaGlutWindowingToolkit.h>
#include <VistaKernel/DisplayManager/VistaDisplayManager.h>
#include <VistaKernel/EventManager/VistaEventManager.h>
#include <VistaKernel/EventManager/VistaSystemEvent.h>
#include <VistaKernel/GraphicsManager/VistaSceneGraph.h>
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

bool Application::Init(VistaSystem* pVistaSystem) {
  // initialize curl - else it will be done in a not-threadsafe-manner in the TileSourcWMS
  cURLpp::initialize();

  mInputManager   = std::make_shared<cs::core::InputManager>();
  mFrameTimings   = std::make_shared<cs::utils::FrameTimings>();
  mGraphicsEngine = std::make_shared<cs::core::GraphicsEngine>(mSettings);
  mTimeControl    = std::make_shared<cs::core::TimeControl>(mSettings);
  mSolarSystem    = std::make_shared<cs::core::SolarSystem>(mTimeControl);
  mGuiManager     = std::make_shared<cs::core::GuiManager>(mSettings, mInputManager, mFrameTimings);

  // clang-format off
  VdfnNodeFactory* pNodeFactory = VdfnNodeFactory::GetSingleton();
  pNodeFactory->SetNodeCreator("ObserverNavigationNode", new ObserverNavigationNodeCreate(mSolarSystem, mInputManager));
  pNodeFactory->SetNodeCreator("DragNavigationNode",     new DragNavigationNodeCreate(mSolarSystem, mInputManager, mTimeControl));
  pNodeFactory->SetNodeCreator("ObserverSyncNode",       new ObserverSyncNodeCreate(mSolarSystem, mTimeControl));
  pNodeFactory->SetNodeCreator("AutoSceneScaleNode",     new AutoSceneScaleNodeCreate(mSolarSystem,  mGraphicsEngine, mTimeControl));
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

  // add some gui callbacks ------------------------------------------------------------------------
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

  // UI updates based on property changes ----------------------------------------------------------
  mTimeControl->pSimulationTime.onChange().connect([this](double val) {
    std::stringstream sstr;
    auto              facet = new boost::posix_time::time_facet();
    facet->format("%d-%b-%Y %H:%M:%S.%f");
    sstr.imbue(std::locale(std::locale::classic(), facet));
    sstr << cs::utils::convert::toBoostTime(val);
    mGuiManager->getHeaderBar()->callJavascript("set_date", sstr.str());
    mGuiManager->getTimeNavigationBar()->callJavascript("set_date", sstr.str());
  });

  mTimeControl->pTimeSpeed.onChange().connect(
      [this](float val) { mGuiManager->getHeaderBar()->callJavascript("set_time_speed", val); });

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

  registerSolarSystemCallbacks();
  registerSideBarCallbacks();
  registerHeaderBarCallbacks();
  registerCalendarCallbacks();

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
    } catch (std::runtime_error const& e) {
      std::cerr << "Error loading plugin " << plugin.first << ": " << e.what() << std::endl;
    }
  }

  return VistaFrameLoop::Init(pVistaSystem);
}

////////////////////////////////////////////////////////////////////////////////////////////////////

void Application::registerSolarSystemCallbacks() {
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

  // show gui notifications ------------------------------------------------------------------------
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
}

////////////////////////////////////////////////////////////////////////////////////////////////////

void Application::registerSideBarCallbacks() {
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
}

////////////////////////////////////////////////////////////////////////////////////////////////////

void Application::registerHeaderBarCallbacks() {
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

  // Time Control -------------------------------------------------------

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

  mGuiManager->getTimeNavigationBar()->registerCallback<double>("add_hours", ([&](double amount) {
    mTimeControl->setTime(mTimeControl->pSimulationTime.get() + 60.0 * 60.0 * amount);
  }));
  mGuiManager->getTimeNavigationBar()->registerCallback<std::string>(
      "set_date", ([this](std::string const& date) {
        mTimeControl->setTime(
            cs::utils::convert::toSpiceTime(boost::posix_time::time_from_string(date)));
      }));
  mGuiManager->getTimeNavigationBar()->registerCallback<std::string>(
      "set_date_direct", ([this](std::string const& date) {
        mTimeControl->setTimeWithoutAnimation(
            cs::utils::convert::toSpiceTime(boost::posix_time::time_from_string(date)));
      }));
  mGuiManager->getTimeNavigationBar()->registerCallback<double>("set_time_speed", ([&](double speed) {
    mTimeControl->setTimeSpeed((float)speed);}));

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
}

////////////////////////////////////////////////////////////////////////////////////////////////////

void Application::registerCalendarCallbacks() {
  mGuiManager->getCalendar()->registerCallback<std::string>(
      "set_date", ([this](std::string const& date) {
        mTimeControl->setTime(
            cs::utils::convert::toSpiceTime(boost::posix_time::time_from_string(date)));
      }));
}

////////////////////////////////////////////////////////////////////////////////////////////////////

void Application::FrameUpdate() {
  ++m_iFrameCount;

  // auto props(
  //     GetVistaSystem()->GetDisplayManager()->GetWindows().begin()->second->GetWindowProperties());
  // int width, height;
  // props->GetSize(width, height);
  // std::cout << width << " " << height << std::endl;

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

  // load plugins ----------------------------------------------------------------------------------
  // Start loading of resources after a short delay to make sure that the loading screen is visible.
  // Then we will draw some frames between each plugin to be able to update the loading screen's
  // status message.
  if (!mLoadedAllPlugins) {
    const int32_t startLoadingAtFrame = 150;
    const int32_t loadingDelayFrames  = 25;

    int32_t nextPluginToLoad = (GetFrameCount() - startLoadingAtFrame) / loadingDelayFrames;

    if (nextPluginToLoad >= 0 && nextPluginToLoad < mPlugins.size()) {
      auto plugin = mPlugins.begin();
      std::advance(plugin, nextPluginToLoad);
      mGuiManager->setLoadingScreenStatus("Loading " + plugin->first + " ...");
    }

    if ((std::max(0, GetFrameCount() - startLoadingAtFrame) % loadingDelayFrames) == 0) {
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
        } catch (std::runtime_error const& e) {
          std::cerr << "Error initializing plugin " << plugin->first << ": " << e.what()
                    << std::endl;
        }

      } else if (pluginToLoad == mPlugins.size()) {

        mLoadedAllPlugins = true;

        // initial observer animation
        // --------------------------------------------------------------
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

  // update CosmoScout VR classes
  {
    cs::utils::FrameTimings::ScopedTimer timer(
        "TimeControl Update", cs::utils::FrameTimings::QueryMode::eCPU);
    mTimeControl->update();
  }

  if (mLoadedAllPlugins) {
    for (auto const& plugin : mPlugins) {
      cs::utils::FrameTimings::ScopedTimer timer(
          plugin.first, cs::utils::FrameTimings::QueryMode::eBoth);

      try {
        plugin.second.mPlugin->update();
      } catch (std::runtime_error const& e) {
        std::cerr << "Error updating plugin " << plugin.first << ": " << e.what() << std::endl;
      }
    }
  }

  {
    cs::utils::FrameTimings::ScopedTimer timer(
        "SolarSystem Update", cs::utils::FrameTimings::QueryMode::eCPU);
    mSolarSystem->update();
  }

  {
    auto sunTransform = mSolarSystem->getSun()->getWorldTransform();
    mGraphicsEngine->setSunDirection(glm::normalize(sunTransform[3].xyz()));
  }

  {
    cs::utils::FrameTimings::ScopedTimer timer(
        "User Interface", cs::utils::FrameTimings::QueryMode::eCPU);

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

  cURLpp::terminate();
}

////////////////////////////////////////////////////////////////////////////////////////////////////
