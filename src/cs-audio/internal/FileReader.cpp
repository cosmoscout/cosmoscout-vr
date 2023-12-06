////////////////////////////////////////////////////////////////////////////////////////////////////
//                               This file is part of CosmoScout VR                               //
////////////////////////////////////////////////////////////////////////////////////////////////////

// SPDX-FileCopyrightText: German Aerospace Center (DLR) <cosmoscout@dlr.de>
// SPDX-License-Identifier: MIT

#include "FileReader.hpp"
#include "BufferManager.hpp"
#include "../logger.hpp"

#include <iostream>
#include <fstream>
#include <cstring>
#include <variant>
#include <algorithm>
#include <cstdlib>
#include <AL/al.h>
#include <AL/alext.h>
#include <sndfile.h>
#include <stdio.h>
#include <cstdint>
#include <limits.h>

namespace cs::audio {

const char * FileReader::getFormatName(ALenum format)
{
  switch(format)
  {
    case AL_FORMAT_MONO8: return "Mono, U8";
    case AL_FORMAT_MONO16: return "Mono, S16";
    case AL_FORMAT_MONO_FLOAT32: return "Mono, Float32";
    case AL_FORMAT_MONO_MULAW: return "Mono, muLaw";
    case AL_FORMAT_MONO_ALAW_EXT: return "Mono, aLaw";
    case AL_FORMAT_MONO_IMA4: return "Mono, IMA4 ADPCM";
    case AL_FORMAT_MONO_MSADPCM_SOFT: return "Mono, MS ADPCM";
    case AL_FORMAT_STEREO8: return "Stereo, U8";
    case AL_FORMAT_STEREO16: return "Stereo, S16";
    case AL_FORMAT_STEREO_FLOAT32: return "Stereo, Float32";
    case AL_FORMAT_STEREO_MULAW: return "Stereo, muLaw";
    case AL_FORMAT_STEREO_ALAW_EXT: return "Stereo, aLaw";
    case AL_FORMAT_STEREO_IMA4: return "Stereo, IMA4 ADPCM";
    case AL_FORMAT_STEREO_MSADPCM_SOFT: return "Stereo, MS ADPCM";
    case AL_FORMAT_QUAD8: return "Quadraphonic, U8";
    case AL_FORMAT_QUAD16: return "Quadraphonic, S16";
    case AL_FORMAT_QUAD32: return "Quadraphonic, Float32";
    case AL_FORMAT_QUAD_MULAW: return "Quadraphonic, muLaw";
    case AL_FORMAT_51CHN8: return "5.1 Surround, U8";
    case AL_FORMAT_51CHN16: return "5.1 Surround, S16";
    case AL_FORMAT_51CHN32: return "5.1 Surround, Float32";
    case AL_FORMAT_51CHN_MULAW: return "5.1 Surround, muLaw";
    case AL_FORMAT_61CHN8: return "6.1 Surround, U8";
    case AL_FORMAT_61CHN16: return "6.1 Surround, S16";
    case AL_FORMAT_61CHN32: return "6.1 Surround, Float32";
    case AL_FORMAT_61CHN_MULAW: return "6.1 Surround, muLaw";
    case AL_FORMAT_71CHN8: return "7.1 Surround, U8";
    case AL_FORMAT_71CHN16: return "7.1 Surround, S16";
    case AL_FORMAT_71CHN32: return "7.1 Surround, Float32";
    case AL_FORMAT_71CHN_MULAW: return "7.1 Surround, muLaw";
    case AL_FORMAT_BFORMAT2D_8: return "B-Format 2D, U8";
    case AL_FORMAT_BFORMAT2D_16: return "B-Format 2D, S16";
    case AL_FORMAT_BFORMAT2D_FLOAT32: return "B-Format 2D, Float32";
    case AL_FORMAT_BFORMAT2D_MULAW: return "B-Format 2D, muLaw";
    case AL_FORMAT_BFORMAT3D_8: return "B-Format 3D, U8";
    case AL_FORMAT_BFORMAT3D_16: return "B-Format 3D, S16";
    case AL_FORMAT_BFORMAT3D_FLOAT32: return "B-Format 3D, Float32";
    case AL_FORMAT_BFORMAT3D_MULAW: return "B-Format 3D, muLaw";
    case AL_FORMAT_UHJ2CHN8_SOFT: return "UHJ 2-channel, U8";
    case AL_FORMAT_UHJ2CHN16_SOFT: return "UHJ 2-channel, S16";
    case AL_FORMAT_UHJ2CHN_FLOAT32_SOFT: return "UHJ 2-channel, Float32";
    case AL_FORMAT_UHJ3CHN8_SOFT: return "UHJ 3-channel, U8";
    case AL_FORMAT_UHJ3CHN16_SOFT: return "UHJ 3-channel, S16";
    case AL_FORMAT_UHJ3CHN_FLOAT32_SOFT: return "UHJ 3-channel, Float32";
    case AL_FORMAT_UHJ4CHN8_SOFT: return "UHJ 4-channel, U8";
    case AL_FORMAT_UHJ4CHN16_SOFT: return "UHJ 4-channel, S16";
    case AL_FORMAT_UHJ4CHN_FLOAT32_SOFT: return "UHJ 4-channel, Float32";
  }
  return "Unknown Format";
}

bool FileReader::readMetaData(std::string fileName, AudioContainer& audioContainer) {
  FormatType      sample_format  = Int16;
  ALint           byteblockalign = 0;
  ALint           splblockalign  = 0;
  ALenum          format;
  SNDFILE*        sndfile;
  SF_INFO         sfinfo;

  /* Open the audio file and check that it's usable. */
  sndfile = sf_open(fileName.c_str(), SFM_READ, &sfinfo);
  if (!sndfile) {
    logger().warn("Could not open audio in {}: {}", fileName, sf_strerror(sndfile));
    return false;
  }
  if (sfinfo.frames < 1) {
    logger().warn("Bad sample count in {}({})", fileName, sfinfo.frames);
    sf_close(sndfile);
    return false;
  }

  /* Detect a suitable format to load. Formats like Vorbis and Opus use float
   * natively, so load as float to avoid clipping when possible. Formats
   * larger than 16-bit can also use float to preserve a bit more precision.
   */
  switch ((sfinfo.format & SF_FORMAT_SUBMASK)) {
  case SF_FORMAT_PCM_24:
  case SF_FORMAT_PCM_32:
  case SF_FORMAT_FLOAT:
  case SF_FORMAT_DOUBLE:
  case SF_FORMAT_VORBIS:
  case SF_FORMAT_OPUS:
  case SF_FORMAT_ALAC_20:
  case SF_FORMAT_ALAC_24:
  case SF_FORMAT_ALAC_32:
  case 0x0080 /*SF_FORMAT_MPEG_LAYER_I*/:
  case 0x0081 /*SF_FORMAT_MPEG_LAYER_II*/:
  case 0x0082 /*SF_FORMAT_MPEG_LAYER_III*/:
    if (alIsExtensionPresent("AL_EXT_FLOAT32"))
      sample_format = Float;
    break;
  case SF_FORMAT_IMA_ADPCM:
    /* ADPCM formats require setting a block alignment as specified in the
     * file, which needs to be read from the wave 'fmt ' chunk manually
     * since libsndfile doesn't provide it in a format-agnostic way.
     */
    if (sfinfo.channels <= 2 && (sfinfo.format & SF_FORMAT_TYPEMASK) == SF_FORMAT_WAV &&
        alIsExtensionPresent("AL_EXT_IMA4") && alIsExtensionPresent("AL_SOFT_block_alignment"))
      sample_format = IMA4;
    break;
  case SF_FORMAT_MS_ADPCM:
    if (sfinfo.channels <= 2 && (sfinfo.format & SF_FORMAT_TYPEMASK) == SF_FORMAT_WAV &&
        alIsExtensionPresent("AL_SOFT_MSADPCM") && alIsExtensionPresent("AL_SOFT_block_alignment"))
      sample_format = MSADPCM;
    break;
  }

  if (sample_format == IMA4 || sample_format == MSADPCM) {
    /* For ADPCM, lookup the wave file's "fmt " chunk, which is a
     * WAVEFORMATEX-based structure for the audio format.
     */
    SF_CHUNK_INFO      inf  = {"fmt ", 4, 0, NULL};
    SF_CHUNK_ITERATOR* iter = sf_get_chunk_iterator(sndfile, &inf);

    /* If there's an issue getting the chunk or block alignment, load as
     * 16-bit and have libsndfile do the conversion.
     */
    if (!iter || sf_get_chunk_size(iter, &inf) != SF_ERR_NO_ERROR || inf.datalen < 14)
      sample_format = Int16;
    else {
      ALubyte* fmtbuf = static_cast<ALubyte*>(calloc(inf.datalen, 1));
      inf.data        = fmtbuf;
      if (sf_get_chunk_data(iter, &inf) != SF_ERR_NO_ERROR)
        sample_format = Int16;
      else {
        /* Read the nBlockAlign field, and convert from bytes- to
         * samples-per-block (verifying it's valid by converting back
         * and comparing to the original value).
         */
        byteblockalign = fmtbuf[12] | (fmtbuf[13] << 8);
        if (sample_format == IMA4) {
          splblockalign = (byteblockalign / sfinfo.channels - 4) / 4 * 8 + 1;
          if (splblockalign < 1 ||
              ((splblockalign - 1) / 2 + 4) * sfinfo.channels != byteblockalign)
            sample_format = Int16;
        } else {
          splblockalign = (byteblockalign / sfinfo.channels - 7) * 2 + 2;
          if (splblockalign < 2 ||
              ((splblockalign - 2) / 2 + 7) * sfinfo.channels != byteblockalign)
            sample_format = Int16;
        }
      }
      free(fmtbuf);
    }
  }

  if (sample_format == Int16) {
    splblockalign  = 1;
    byteblockalign = sfinfo.channels * 2;
  } else if (sample_format == Float) {
    splblockalign  = 1;
    byteblockalign = sfinfo.channels * 4;
  }

  /* Figure out the OpenAL format from the file and desired sample type. */
  format = AL_NONE;
  if (sfinfo.channels == 1) {
    if (sample_format == Int16)
      format = AL_FORMAT_MONO16;
    else if (sample_format == Float)
      format = AL_FORMAT_MONO_FLOAT32;
    else if (sample_format == IMA4)
      format = AL_FORMAT_MONO_IMA4;
    else if (sample_format == MSADPCM)
      format = AL_FORMAT_MONO_MSADPCM_SOFT;
  } else if (sfinfo.channels == 2) {
    if (sample_format == Int16)
      format = AL_FORMAT_STEREO16;
    else if (sample_format == Float)
      format = AL_FORMAT_STEREO_FLOAT32;
    else if (sample_format == IMA4)
      format = AL_FORMAT_STEREO_IMA4;
    else if (sample_format == MSADPCM)
      format = AL_FORMAT_STEREO_MSADPCM_SOFT;
  } else if (sfinfo.channels == 3) {
    if (sf_command(sndfile, SFC_WAVEX_GET_AMBISONIC, NULL, 0) == SF_AMBISONIC_B_FORMAT) {
      if (sample_format == Int16)
        format = AL_FORMAT_BFORMAT2D_16;
      else if (sample_format == Float)
        format = AL_FORMAT_BFORMAT2D_FLOAT32;
    }
  } else if (sfinfo.channels == 4) {
    if (sf_command(sndfile, SFC_WAVEX_GET_AMBISONIC, NULL, 0) == SF_AMBISONIC_B_FORMAT) {
      if (sample_format == Int16)
        format = AL_FORMAT_BFORMAT3D_16;
      else if (sample_format == Float)
        format = AL_FORMAT_BFORMAT3D_FLOAT32;
    }
  }
  if (!format) {
    logger().warn("Unsupported channel count: {}", sfinfo.channels);
    sf_close(sndfile);
    return false;
  }

  if (sfinfo.frames / splblockalign > (sf_count_t)(INT_MAX / byteblockalign)) {
    logger().warn("Too many samples in {} ({})", fileName, sfinfo.frames);
    sf_close(sndfile);
    return false;
  }

  audioContainer.format = format;
  audioContainer.formatType = sample_format;
  audioContainer.sfInfo = sfinfo; 
  audioContainer.splblockalign = splblockalign;
  audioContainer.byteblockalign = byteblockalign;
  audioContainer.sndFile = sndfile;
  return true;
}

////////////////////////////////////////////////////////////////////////////////////////////////////

bool FileReader::loadFile(std::string fileName, AudioContainer& audioContainer) {
  
  if (!readMetaData(fileName, audioContainer)) {
    return false;
  }

  /* Decode the whole audio file to a buffer. */
  sf_count_t num_frames;
  switch (audioContainer.formatType) {
    case Int16:
      audioContainer.audioData =
        std::vector<short>((size_t)(audioContainer.sfInfo.frames / audioContainer.splblockalign * audioContainer.byteblockalign));
      num_frames = sf_readf_short(
        audioContainer.sndFile, 
        std::get<std::vector<short>>(audioContainer.audioData).data(), 
        audioContainer.sfInfo.frames);
      break;

    case Float:
      audioContainer.audioData =
        std::vector<float>((size_t)(audioContainer.sfInfo.frames / audioContainer.splblockalign * audioContainer.byteblockalign));
      num_frames = sf_readf_float(
        audioContainer.sndFile, 
        std::get<std::vector<float>>(audioContainer.audioData).data(),
        audioContainer.sfInfo.frames);
      break;

    default:
      audioContainer.audioData =
        std::vector<int>((size_t)(audioContainer.sfInfo.frames / audioContainer.splblockalign * audioContainer.byteblockalign));
      sf_count_t count = audioContainer.sfInfo.frames / audioContainer.splblockalign * audioContainer.byteblockalign;
      num_frames =
        sf_read_raw(audioContainer.sndFile, 
        std::get<std::vector<int>>(audioContainer.audioData).data(), 
        count);
      if (num_frames > 0) {
        num_frames = num_frames / audioContainer.byteblockalign * audioContainer.splblockalign;    
      }
  }

  if (num_frames < 1) {
    audioContainer.reset();
    logger().warn("Failed to read samples in {} ({})", fileName, num_frames);
    return false;
  }

  audioContainer.size = (ALsizei)(num_frames / audioContainer.splblockalign * audioContainer.byteblockalign);
  sf_close(audioContainer.sndFile);
  return true;
}

////////////////////////////////////////////////////////////////////////////////////////////////////

bool FileReader::openStream(std::string fileName, AudioContainerStreaming& audioContainer) {

  audioContainer.reset();
  if (!readMetaData(fileName, audioContainer)) {
    logger().warn("readMetaData() failed");
    return false;
  }

  audioContainer.blockCount = audioContainer.sfInfo.samplerate / audioContainer.splblockalign;
  audioContainer.blockCount = audioContainer.blockCount * audioContainer.bufferLength / 1000;

  switch (audioContainer.formatType) {
    case Int16:
      audioContainer.audioData = std::vector<short>((size_t)(audioContainer.blockCount * audioContainer.byteblockalign));
      break;
    case Float:
      audioContainer.audioData = std::vector<float>((size_t)(audioContainer.blockCount * audioContainer.byteblockalign));
      break;
    default:
      audioContainer.audioData = std::vector<int>((size_t)(audioContainer.blockCount * audioContainer.byteblockalign));
  }

  return true;
}

////////////////////////////////////////////////////////////////////////////////////////////////////

bool FileReader::getNextStreamBlock(AudioContainerStreaming& audioContainer) {

  sf_count_t slen;
  switch (audioContainer.formatType) {
    case Int16:
      slen = sf_readf_short(audioContainer.sndFile, std::get<std::vector<short>>(audioContainer.audioData).data(),
        audioContainer.blockCount * audioContainer.splblockalign);

      if (slen < 1) {
        sf_seek(audioContainer.sndFile, 0, SEEK_SET);

        if (audioContainer.isLooping) {
          return getNextStreamBlock(audioContainer);
        } else {
          return false;
        }
      }
      slen *= audioContainer.byteblockalign;
      break;

    case Float:
      slen = sf_readf_float(audioContainer.sndFile, std::get<std::vector<float>>(audioContainer.audioData).data(),
        audioContainer.blockCount * audioContainer.splblockalign);
      if (slen < 1) {
        sf_seek(audioContainer.sndFile, 0, SEEK_SET);

        if (audioContainer.isLooping) {
          return getNextStreamBlock(audioContainer);
        } else {
          return false;
        }
      }
      slen *= audioContainer.byteblockalign;
      break;

    default:
      slen = sf_read_raw(audioContainer.sndFile, std::get<std::vector<int>>(audioContainer.audioData).data(),
        audioContainer.blockCount * audioContainer.splblockalign);
      if (slen > 0)
        slen -= slen % audioContainer.byteblockalign;
      if (slen < 1)
        sf_seek(audioContainer.sndFile, 0, SEEK_SET);
        
        if (audioContainer.isLooping) {
          return getNextStreamBlock(audioContainer);
        } else {
          return false;
        }
  }
  audioContainer.bufferSize = slen;
  return true;
}

} // namespace cs::audio