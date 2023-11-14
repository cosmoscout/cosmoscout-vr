////////////////////////////////////////////////////////////////////////////////////////////////////
//                               This file is part of CosmoScout VR                               //
////////////////////////////////////////////////////////////////////////////////////////////////////

// SPDX-FileCopyrightText: German Aerospace Center (DLR) <cosmoscout@dlr.de>
// SPDX-License-Identifier: MIT

#ifndef CS_AUDIO_PS_SPHERE_SOURCE_HPP
#define CS_AUDIO_PS_SPHERE_SOURCE_HPP

#include "cs_audio_export.hpp"
#include "ProcessingStep.hpp"
#include "SpatializationUtils.hpp"
#include "../internal/SourceBase.hpp"
#include <memory>
#include <AL/al.h>
#include <glm/fwd.hpp>
#include <chrono>

namespace cs::audio {
/*
The SphereSpatialization_PS is a spatialization processing step with which lets you define a position 
as a 3D sphere in space. If the observer is inside the sphere you will hear the source at full volume 
without spatialization. If the sphere is large and the observer leaves the sphere you will notice that 
the source will most probably cut off immediately. This is because once the observer is outside the sphere 
the source gets positioned at the center of the sphere and due to the distance attenuation the volume drops 
to zero. If this is not the behaviour you want, you can use the DistanceModel processing step and set the 
fallOffStart to the sphere radius. This will enable the distance attenuation only at the edge of the sphere.
This processing step will also automatically compute the velocity of a source and the observer. 
The position must be specified relative to the observer.
---------------------------------------------------------
Name          Type          Range       Description
---------------------------------------------------------
position      glm::dvec3                Position of a source relative to the observer.
sourceRadius  double        0.0 -       Radius of the sphere.
---------------------------------------------------------
*/
class CS_AUDIO_EXPORT SphereSpatialization_PS 
  : public ProcessingStep
  , public SpatializationUtils {
 public:

  static std::shared_ptr<ProcessingStep> create();

  void process(std::shared_ptr<SourceBase> source, 
   std::shared_ptr<std::map<std::string, std::any>> settings,
   std::shared_ptr<std::vector<std::string>> failedSettings);

  bool requiresUpdate() const;

  void update();

 private:

  SphereSpatialization_PS();
  bool processPosition(ALuint openAlId, std::any position);
  bool processRadius(ALuint openAlId, std::any sourceRadius);
  bool processSpatialization(std::shared_ptr<SourceBase> source, std::any position,
    std::any sourceRadius);
  bool resetSpatialization(ALuint openAlId);
};

} // namespace cs::audio

#endif // CS_AUDIO_PS_SPHERE_SOURCE_HPP
