[Setup]
AppName                           = CosmoScout VR - VESTEC Version
AppVersion                        = 0.1.0
VersionInfoVersion                = 0.1.0.0
ArchitecturesAllowed              = x64
ArchitecturesInstallIn64BitMode   = x64
Compression                       = lzma2
DiskSpanning                      = true
DefaultDirName                    = {userpf}\CosmoScout VR
DefaultGroupName                  = CosmoScout VR
DisableWelcomePage                = no
OutputDir                         = D:\

OutputBaseFilename                = CosmoScout VR Setup
PrivilegesRequired                = lowest
SetupIconFile                     = resources/icons/icon.ico
SolidCompression                  = yes
UninstallDisplayIcon              = {app}\cosmoscout.exe

;Welcome screen
[Messages]
WelcomeLabel2=Please do not try to install CosmoScout VR to a directory which requires administrator privileges for writing. 

;Creating all directories needed for the installation
[Dirs]
Name: "{app}\bin"
Name: "{app}\bin\map-cache"
Name: "{app}\bin\locales"
Name: "{app}\bin\proj6"
Name: "{app}\lib"
Name: "{app}\share\"
Name: "{app}\share\config"
Name: "{app}\share\plugins"
Name: "{app}\share\config\vista"
Name: "{app}\share\resources"
Name: "{app}\share\doc"

; Collection of files and directories to copy
[Files] 
Source: "..\install\windows-Release\bin\*"; 		      Excludes: "*.cmd,*.dot,star_cache.dat"; DestDir: "{app}\bin"; Flags: recursesubdirs;
Source: "..\install\windows-Release\lib\*";           Excludes: "*.lib"; DestDir: "{app}\lib"; Flags: recursesubdirs;
Source: "..\install\windows-Release\share\*"; 		    Excludes: "*.lib"; DestDir: "{app}\share"; Flags: recursesubdirs; 
Source: "..\install\windows-Release\docs\*";          DestDir: "{app}\share\docs"; Flags: recursesubdirs;   

;Define some icons
[Icons]
Name: "{userdesktop}\CosmoScout VR"; Filename: "{app}\bin\start.bat"; WorkingDir: "{app}\bin";  IconFilename: "{app}\share\resources\icons\icon.ico"
Name: "{group}\CosmoScout VR";         Filename: "{app}\bin\start.bat"; WorkingDir: "{app}\bin";  IconFilename: "{app}\share\resources\icons\icon.ico"
Name: "{group}\{cm:UninstallProgram,CosmoScout VR}"; Filename: "{uninstallexe}"

;Run
[Run]
Filename: "{app}\bin\start.bat"; Description: "Launch CosmoScout VR"; Flags: postinstall nowait skipifsilent