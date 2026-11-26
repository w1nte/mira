; Inno Setup script: packs the PyInstaller folder (dist\vision-tagger) into a setup program.
;
;   iscc /DAppVersion=0.1.0 /DFlavor=gpu packaging\windows\installer.iss
;
; Installs per user (no administrator rights) into %LOCALAPPDATA%\Programs\Mira (an update of
; an installation from before the rename stays in its folder).
; The library (config, database, thumbnails) lives in %LOCALAPPDATA%\vision-tagger and is kept
; on uninstall and update; the photos themselves are never touched.

#ifndef AppVersion
  #define AppVersion "0.0.0"
#endif
#ifndef Flavor
  #define Flavor "gpu"
#endif
#define AppName "Mira"
#define AppExe "Mira.exe"

[Setup]
AppId={{BDA0ECA7-54BD-4B4D-82BE-1592C783603F}
AppName={#AppName}
AppVersion={#AppVersion}
AppVerName={#AppName} {#AppVersion}
AppPublisher={#AppName}
AppPublisherURL=https://github.com/w1nte/vision-tagger
DefaultDirName={autopf}\{#AppName}
DefaultGroupName={#AppName}
DisableProgramGroupPage=yes
PrivilegesRequired=lowest
PrivilegesRequiredOverridesAllowed=dialog
ArchitecturesAllowed=x64compatible
ArchitecturesInstallIn64BitMode=x64compatible
MinVersion=10.0
OutputDir=..\..\dist
OutputBaseFilename={#AppName}-setup-{#AppVersion}-{#Flavor}
SetupIconFile=build\vision-tagger.ico
UninstallDisplayIcon={app}\{#AppExe}
UninstallDisplayName={#AppName}
WizardStyle=modern
Compression=lzma2/max
SolidCompression=yes
LZMAUseSeparateProcess=yes
CloseApplications=yes
#if Flavor == "gpu"
; the CUDA and cuDNN libraries make the GPU build too big for a single setup file
DiskSpanning=yes
DiskSliceSize=max
#endif

[Languages]
Name: "en"; MessagesFile: "compiler:Default.isl"
Name: "de"; MessagesFile: "compiler:Languages\German.isl"
Name: "fr"; MessagesFile: "compiler:Languages\French.isl"
Name: "es"; MessagesFile: "compiler:Languages\Spanish.isl"
Name: "it"; MessagesFile: "compiler:Languages\Italian.isl"
Name: "pt"; MessagesFile: "compiler:Languages\Portuguese.isl"
Name: "ptbr"; MessagesFile: "compiler:Languages\BrazilianPortuguese.isl"
Name: "nl"; MessagesFile: "compiler:Languages\Dutch.isl"
Name: "pl"; MessagesFile: "compiler:Languages\Polish.isl"
Name: "ru"; MessagesFile: "compiler:Languages\Russian.isl"
Name: "ja"; MessagesFile: "compiler:Languages\Japanese.isl"

[Tasks]
Name: "desktopicon"; Description: "{cm:CreateDesktopIcon}"; GroupDescription: "{cm:AdditionalIcons}"; Flags: unchecked

[InstallDelete]
; files of the previous version that this one no longer has must not linger
Type: filesandordirs; Name: "{app}\_internal"
; the app and its shortcuts were called vision-tagger before
Type: files; Name: "{app}\vision-tagger.exe"
Type: files; Name: "{autoprograms}\vision-tagger.lnk"
Type: files; Name: "{autodesktop}\vision-tagger.lnk"

[Files]
Source: "..\..\dist\vision-tagger\*"; DestDir: "{app}"; Flags: ignoreversion recursesubdirs createallsubdirs

[Icons]
Name: "{autoprograms}\{#AppName}"; Filename: "{app}\{#AppExe}"
Name: "{autodesktop}\{#AppName}"; Filename: "{app}\{#AppExe}"; Tasks: desktopicon

[Run]
Filename: "{app}\{#AppExe}"; Description: "{cm:LaunchProgram,{#AppName}}"; Flags: nowait postinstall skipifsilent
