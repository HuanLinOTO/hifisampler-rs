; HiFiSampler Inno Setup Script
; ─────────────────────────────────────────────────────────────
; Local build:
;   iscc /DSourceDir=..\dist\hifisampler-windows-x64-cpu setup.iss
;
; CI build:
;   iscc /DSourceDir=D:\a\repo\dist\hifisampler-windows-x64-directml ^
;        /DVariant=directml /DMyAppVersion=0.2.0 setup.iss
;
; Requires Inno Setup 6.x — https://jrsoftware.org/isinfo.php
; ─────────────────────────────────────────────────────────────

; ── Defaults (overridable via /D on command line) ──
#ifndef SourceDir
  #define SourceDir "..\dist\hifisampler-windows-x64-cpu"
#endif
#ifndef Variant
  #define Variant "cpu"
#endif
#define MyAppName "HiFiSampler"
#ifndef MyAppVersion
  #define MyAppVersion "0.1.0"
#endif
#define MyAppPublisher "OpenHachimi"
#define MyAppURL "https://github.com/openhachimi/hifisampler"
#define MyAppExeName "hifisampler-server.exe"

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
; LICENSE is optional — only set if present
#if FileExists(AddBackslash(SourceDir) + "LICENSE")
LicenseFile={#SourceDir}\LICENSE
#endif
OutputDir=output
OutputBaseFilename=HiFiSampler-{#MyAppVersion}-{#Variant}-Setup
Compression=lzma2/ultra64
SolidCompression=yes
WizardStyle=modern
ArchitecturesAllowed=x64compatible
ArchitecturesInstallIn64BitMode=x64compatible
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
; ── Binaries ──
Source: "{#SourceDir}\hifisampler-server.exe"; DestDir: "{app}"; Flags: ignoreversion
Source: "{#SourceDir}\hifisampler.exe"; DestDir: "{app}"; Flags: ignoreversion

; ── ONNX Runtime DLLs (varies per EP variant) ──
Source: "{#SourceDir}\*.dll"; DestDir: "{app}"; Flags: ignoreversion skipifsourcedoesntexist

; ── Config ──
Source: "{#SourceDir}\config.default.yaml"; DestDir: "{app}"; DestName: "config.yaml"; Flags: onlyifdoesntexist uninsneveruninstall
Source: "{#SourceDir}\config.default.yaml"; DestDir: "{app}"; Flags: ignoreversion

; ── WebUI ──
Source: "{#SourceDir}\webui\*"; DestDir: "{app}\webui"; Flags: ignoreversion recursesubdirs createallsubdirs skipifsourcedoesntexist

; ── Models ──
Source: "{#SourceDir}\models\*"; DestDir: "{app}\models"; Flags: ignoreversion recursesubdirs createallsubdirs skipifsourcedoesntexist

; ── Docs ──
Source: "{#SourceDir}\README.md"; DestDir: "{app}"; Flags: ignoreversion skipifsourcedoesntexist
Source: "{#SourceDir}\LICENSE"; DestDir: "{app}"; Flags: ignoreversion skipifsourcedoesntexist

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
    '请浏览并选择 OpenUTAU 的 Resamplers 目录。' + #13#10 +
    '桥接程序将被复制到此处，以便 OpenUTAU 调用 HiFiSampler 引擎。' + #13#10 + #13#10 +
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
      Dest := AddBackslash(OpenUtauDirPage.Values[0]) + 'hifisampler.exe';
      if DirExists(OpenUtauDirPage.Values[0]) then
      begin
        FileCopy(Src, Dest, False);
        MsgBox('桥接程序已安装到：' + #13#10 + Dest + #13#10#13#10 +
          '请重启 OpenUTAU，切换到 Classic 模式，' + #13#10 +
          '在 Resampler 列表中选择 hifisampler.exe。', mbInformation, MB_OK);
      end
      else
      begin
        if MsgBox('Resamplers 目录不存在：' + #13#10 + OpenUtauDirPage.Values[0] + #13#10#13#10 +
          '是否创建该目录并安装？', mbConfirmation, MB_YESNO) = IDYES then
        begin
          ForceDirectories(OpenUtauDirPage.Values[0]);
          FileCopy(Src, Dest, False);
          MsgBox('桥接程序已安装到：' + #13#10 + Dest, mbInformation, MB_OK);
        end;
      end;
    end;
  end;
end;
