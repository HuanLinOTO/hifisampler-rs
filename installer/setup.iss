; HiFiSampler Inno Setup Script
; Build with: iscc setup.iss
; Requires Inno Setup 6.x — https://jrsoftware.org/isinfo.php

#define MyAppName "HiFiSampler"
#define MyAppVersion "0.1.0"
#define MyAppPublisher "OpenHachimi"
#define MyAppURL "https://github.com/openhachimi/hifisampler"
#define MyAppExeName "hifisampler-server.exe"

; ── Paths to build output (adjust if needed) ──
; Assumes this script is run from hifisampler-rs/installer/
; and the release build is at hifisampler-rs/target/x86_64-pc-windows-msvc/release/
#define BuildDir "..\target\x86_64-pc-windows-msvc\release"
#define RootDir ".."

[Setup]
AppId={{B5E3A7D0-8C1F-4F9A-A2D1-HIFISAMPLER01}
AppName={#MyAppName}
AppVersion={#MyAppVersion}
AppPublisher={#MyAppPublisher}
AppPublisherURL={#MyAppURL}
AppSupportURL={#MyAppURL}/issues
DefaultDirName={autopf}\{#MyAppName}
DefaultGroupName={#MyAppName}
AllowNoIcons=yes
LicenseFile={#RootDir}\LICENSE
OutputDir=output
OutputBaseFilename=HiFiSampler-{#MyAppVersion}-Setup
Compression=lzma2/ultra64
SolidCompression=yes
WizardStyle=modern
ArchitecturesAllowed=x64compatible
ArchitecturesInstallIn64BitMode=x64compatible
SetupIconFile=
UninstallDisplayIcon={app}\{#MyAppExeName}
PrivilegesRequired=lowest
PrivilegesRequiredOverridesAllowed=dialog

[Languages]
Name: "chinesesimplified"; MessagesFile: "compiler:Languages\ChineseSimplified.isl"
Name: "english"; MessagesFile: "compiler:Default.isl"

[Tasks]
Name: "desktopicon"; Description: "创建桌面快捷方式"; GroupDescription: "快捷方式:"
Name: "installbridge"; Description: "自动安装桥接程序到 OpenUTAU Resamplers"; GroupDescription: "OpenUTAU 集成:"; Flags: unchecked

[Files]
; Server & Bridge binaries
Source: "{#BuildDir}\hifisampler-server.exe"; DestDir: "{app}"; Flags: ignoreversion
Source: "{#BuildDir}\hifisampler.exe"; DestDir: "{app}"; Flags: ignoreversion

; Config
Source: "{#RootDir}\config.default.yaml"; DestDir: "{app}"; DestName: "config.yaml"; Flags: onlyifdoesntexist
Source: "{#RootDir}\config.default.yaml"; DestDir: "{app}"; Flags: ignoreversion

; WebUI
Source: "{#RootDir}\webui\*"; DestDir: "{app}\webui"; Flags: ignoreversion recursesubdirs createallsubdirs

; Models
Source: "{#RootDir}\models\vocoder\model.onnx"; DestDir: "{app}\models\vocoder"; Flags: ignoreversion; Check: FileExists(ExpandConstant('{#RootDir}\models\vocoder\model.onnx'))
Source: "{#RootDir}\models\hnsep\model.onnx"; DestDir: "{app}\models\hnsep"; Flags: ignoreversion; Check: FileExists(ExpandConstant('{#RootDir}\models\hnsep\model.onnx'))

; Docs
Source: "{#RootDir}\README.md"; DestDir: "{app}"; Flags: ignoreversion
Source: "{#RootDir}\LICENSE"; DestDir: "{app}"; Flags: ignoreversion

[Icons]
Name: "{group}\HiFiSampler Server"; Filename: "{app}\{#MyAppExeName}"; WorkingDir: "{app}"
Name: "{group}\HiFiSampler WebUI"; Filename: "http://127.0.0.1:8572/ui/"
Name: "{group}\卸载 HiFiSampler"; Filename: "{uninstallexe}"
Name: "{autodesktop}\HiFiSampler"; Filename: "{app}\{#MyAppExeName}"; WorkingDir: "{app}"; Tasks: desktopicon

[Run]
Filename: "{app}\{#MyAppExeName}"; Description: "启动 HiFiSampler Server"; Flags: nowait postinstall skipifsilent

[Code]
var
  OpenUtauDirPage: TInputDirWizardPage;

procedure InitializeWizard;
begin
  OpenUtauDirPage := CreateInputDirPage(wpSelectTasks,
    'OpenUTAU Resamplers 目录',
    '选择 OpenUTAU 的 Resamplers 文件夹',
    '请浏览并选择 OpenUTAU 的 Resamplers 目录。桥接程序将被复制到此处。' + #13#10 +
    '（通常位于 C:\Users\你的用户名\OpenUtau\Resamplers）',
    False, '');
  OpenUtauDirPage.Add('');
  OpenUtauDirPage.Values[0] := ExpandConstant('{userappdata}\OpenUtau\Resamplers');
end;

function ShouldSkipPage(PageID: Integer): Boolean;
begin
  Result := False;
  if PageID = OpenUtauDirPage.ID then
    Result := not WizardIsTaskSelected('installbridge');
end;

procedure CurStepChanged(CurStep: TSetupStep);
var
  Src, Dest: String;
begin
  if CurStep = ssPostInstall then
  begin
    if WizardIsTaskSelected('installbridge') then
    begin
      Src := ExpandConstant('{app}\hifisampler.exe');
      Dest := OpenUtauDirPage.Values[0] + '\hifisampler.exe';
      if DirExists(OpenUtauDirPage.Values[0]) then
      begin
        FileCopy(Src, Dest, False);
        MsgBox('桥接程序已安装到：' + #13#10 + Dest + #13#10#13#10 +
          '请重启 OpenUTAU，切换到 Classic 模式，' + #13#10 +
          '在 Resampler 列表中选择 hifisampler.exe。', mbInformation, MB_OK);
      end
      else
        MsgBox('OpenUTAU Resamplers 目录不存在：' + #13#10 +
          OpenUtauDirPage.Values[0] + #13#10 +
          '请手动将 hifisampler.exe 复制到该目录。', mbError, MB_OK);
    end;
  end;
end;
