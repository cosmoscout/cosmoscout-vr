////////////////////////////////////////////////////////////////////////////////////////////////////
//                               This file is part of CosmoScout VR                               //
////////////////////////////////////////////////////////////////////////////////////////////////////

// SPDX-FileCopyrightText: German Aerospace Center (DLR) <cosmoscout@dlr.de>
// SPDX-License-Identifier: MIT

#include "FileReader.hpp"

#include <iostream>
#include <fstream>
#include <cstring>
#include <AL/al.h>

namespace cs::audio {

char* FileReader::loadWAV(const char* fn, int& chan, int& samplerate, int& bps, int& size, unsigned int& format)
{
    char fileBuffer[4];
    std::ifstream in(fn, std::ios::binary);
    in.read(fileBuffer, 4);
    if (strncmp(fileBuffer, "RIFF", 4) != 0)
    {
        std::cout << "this is not a valid WAVE file" << std::endl;
        return NULL;
    }
    in.read(fileBuffer, 4);
    in.read(fileBuffer, 4);      //WAVE
    in.read(fileBuffer, 4);      //fmt
    in.read(fileBuffer, 4);      //16
    in.read(fileBuffer, 2);      //1
    in.read(fileBuffer, 2);
    chan = convertToInt(fileBuffer, 2);
    in.read(fileBuffer, 4);
    samplerate = convertToInt(fileBuffer, 4);
    in.read(fileBuffer, 4);
    in.read(fileBuffer, 2);
    in.read(fileBuffer, 2);
    bps = convertToInt(fileBuffer, 2);
    in.read(fileBuffer, 4);      //data
    in.read(fileBuffer, 4);
    size = convertToInt(fileBuffer, 4);
    char* data = new char[size];
    in.read(data, size);

	if (chan == 1)
	{
		if (bps == 8)
		{
			format = AL_FORMAT_MONO8;
		}
		else {
			format = AL_FORMAT_MONO16;
		}
	}
	else {
		if (bps == 8)
		{
			format = AL_FORMAT_STEREO8;
		}
		else {
			format = AL_FORMAT_STEREO16;
		}
	}

    return data;
}

int FileReader::convertToInt(char* buffer, int len)
{
    int a = 0;
    if (!isBigEndian())
        for (int i = 0; i < len; i++)
            ((char*)&a)[i] = buffer[i];
    else
        for (int i = 0; i < len; i++)
            ((char*)&a)[3 - i] = buffer[i];
    return a;
}

bool FileReader::isBigEndian()
{
    int a = 1;
    return !((char*)&a)[0];
}

} // namespace cs::audio