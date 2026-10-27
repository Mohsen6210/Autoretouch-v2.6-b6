; Run the build first, then compile this file with Inno Setup (jrsoftware.org)
[Setup]
AppName=Auto Retouch
AppVersion=2.0
DefaultDirName={autopf}\AutoRetouch
DefaultGroupName=Auto Retouch
OutputBaseFilename=AutoRetouch_Setup
OutputDir=Output
Compression=lzma2
SolidCompression=yes
[Files]
Source: "dist\AutoRetouch\*"; DestDir: "{app}"; Flags: recursesubdirs
[Icons]
Name: "{group}\Auto Retouch"; Filename: "{app}\AutoRetouch.exe"
Name: "{autodesktop}\Auto Retouch"; Filename: "{app}\AutoRetouch.exe"
