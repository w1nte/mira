; Inno Setup script: packs the PyInstaller folder (dist\vision-tagger) into a setup program in
; dist\installer.
;
;   iscc /DAppVersion=1.2.3 /DFlavor=gpu packaging\windows\installer.iss
;
; Installs per user (no administrator rights) into %LOCALAPPDATA%\Programs\Mira (an update of
; an installation from before the rename stays in its folder).
; The library (config, database, thumbnails) lives in %LOCALAPPDATA%\vision-tagger and is kept
; on uninstall and update; the photos themselves are never touched.
;
; Over an installed Mira (same AppId) this is an update: a page says so, a running Mira is asked
; to quit gracefully first (vision-tagger-cli quit-app; the Restart Manager remains the fallback),
; and afterwards the library's database is migrated (vision-tagger-cli migrate --desktop, with a
; backup). The migration also runs on a fresh install that finds a library kept from before.
; This file is UTF-8 with a byte order mark, for the translated messages.

#ifndef AppVersion
  #define AppVersion "0.0.0"
#endif
#ifndef Flavor
  #define Flavor "gpu"
#endif
#define AppName "Mira"
#define AppExe "Mira.exe"
#define CliExe "vision-tagger-cli.exe"

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
OutputDir=..\..\dist\installer
OutputBaseFilename={#AppName}-setup-{#AppVersion}-{#Flavor}
SetupIconFile=..\build\vision-tagger.ico
UninstallDisplayIcon={app}\{#AppExe}
UninstallDisplayName={#AppName}
WizardStyle=modern
Compression=lzma2/max
SolidCompression=yes
LZMAUseSeparateProcess=yes
; the fallback when a running Mira does not quit on request (see PrepareToInstall)
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

[CustomMessages]
; %1 = installed version, %2 = this version (UpdateCaption, UpdateSubtitle, UpdateMemo);
; %1 = exit code, %2 = default data folder (MigrationFailed)
en.UpdateCaption=Update Mira %1 → %2
en.UpdateTitle=Update Mira
en.UpdateSubtitle=Mira %1 is installed. This setup updates it to version %2.
en.UpdateText=The program files are replaced. Your library (settings, database, thumbnails) is kept, and its database is migrated to the new version; a backup copy is made first.%n%nIf Mira is running, it is closed first. A running pipeline is paused and continues on the next start.
en.UpdateMemoTitle=Update:
en.UpdateMemo=From %1 to %2; the library is kept and its database migrated (with a backup copy)
en.ClosingApp=Closing Mira...
en.MigratingLibrary=Updating the library's database...
en.MigrationFailed=The library's database could not be updated (error %1). Mira tries again when it starts.%n%nDetails: logs\migrate.log in the library's data folder (by default %2).
en.DatabaseTooNew=The library's database is from a newer version of Mira, which this version cannot open. Install the newer version again, or restore a backup copy of the database.

de.UpdateCaption=Mira aktualisieren: %1 → %2
de.UpdateTitle=Mira aktualisieren
de.UpdateSubtitle=Mira %1 ist installiert. Dieses Setup aktualisiert es auf Version %2.
de.UpdateText=Die Programmdateien werden ersetzt. Ihre Bibliothek (Einstellungen, Datenbank, Vorschaubilder) bleibt erhalten, und ihre Datenbank wird auf die neue Version migriert; vorher wird eine Sicherungskopie angelegt.%n%nLäuft Mira, wird es zuerst beendet. Eine laufende Pipeline wird pausiert und beim nächsten Start fortgesetzt.
de.UpdateMemoTitle=Aktualisierung:
de.UpdateMemo=Von %1 auf %2; die Bibliothek bleibt erhalten, ihre Datenbank wird migriert (mit Sicherungskopie)
de.ClosingApp=Mira wird beendet...
de.MigratingLibrary=Die Datenbank der Bibliothek wird aktualisiert...
de.MigrationFailed=Die Datenbank der Bibliothek konnte nicht aktualisiert werden (Fehler %1). Mira versucht es beim nächsten Start erneut.%n%nDetails: logs\migrate.log im Datenordner der Bibliothek (standardmäßig %2).
de.DatabaseTooNew=Die Datenbank der Bibliothek stammt von einer neueren Version von Mira, die diese Version nicht öffnen kann. Installieren Sie die neuere Version erneut, oder stellen Sie eine Sicherungskopie der Datenbank wieder her.

fr.UpdateCaption=Mise à jour de Mira : %1 → %2
fr.UpdateTitle=Mettre à jour Mira
fr.UpdateSubtitle=Mira %1 est installé. Ce programme le met à jour vers la version %2.
fr.UpdateText=Les fichiers du programme sont remplacés. Votre bibliothèque (paramètres, base de données, miniatures) est conservée et sa base de données est migrée vers la nouvelle version ; une copie de sauvegarde est faite auparavant.%n%nSi Mira est ouvert, il est d'abord fermé. Un pipeline en cours est mis en pause et reprend au prochain démarrage.
fr.UpdateMemoTitle=Mise à jour :
fr.UpdateMemo=De %1 vers %2 ; la bibliothèque est conservée et sa base de données migrée (avec une copie de sauvegarde)
fr.ClosingApp=Fermeture de Mira...
fr.MigratingLibrary=Mise à jour de la base de données de la bibliothèque...
fr.MigrationFailed=La base de données de la bibliothèque n'a pas pu être mise à jour (erreur %1). Mira réessaiera au prochain démarrage.%n%nDétails : logs\migrate.log dans le dossier de données de la bibliothèque (par défaut %2).
fr.DatabaseTooNew=La base de données de la bibliothèque provient d'une version plus récente de Mira, que cette version ne peut pas ouvrir. Réinstallez la version plus récente ou restaurez une copie de sauvegarde de la base de données.

es.UpdateCaption=Actualizar Mira: %1 → %2
es.UpdateTitle=Actualizar Mira
es.UpdateSubtitle=Mira %1 está instalado. Este programa lo actualiza a la versión %2.
es.UpdateText=Se reemplazan los archivos del programa. Su biblioteca (ajustes, base de datos, miniaturas) se conserva y su base de datos se migra a la nueva versión; antes se hace una copia de seguridad.%n%nSi Mira se está ejecutando, se cierra primero. Un pipeline en curso se pausa y continúa en el próximo inicio.
es.UpdateMemoTitle=Actualización:
es.UpdateMemo=De %1 a %2; la biblioteca se conserva y su base de datos se migra (con copia de seguridad)
es.ClosingApp=Cerrando Mira...
es.MigratingLibrary=Actualizando la base de datos de la biblioteca...
es.MigrationFailed=No se pudo actualizar la base de datos de la biblioteca (error %1). Mira lo intentará de nuevo al iniciarse.%n%nDetalles: logs\migrate.log en la carpeta de datos de la biblioteca (por defecto %2).
es.DatabaseTooNew=La base de datos de la biblioteca es de una versión más reciente de Mira, que esta versión no puede abrir. Vuelva a instalar la versión más reciente o restaure una copia de seguridad de la base de datos.

it.UpdateCaption=Aggiornamento di Mira: %1 → %2
it.UpdateTitle=Aggiorna Mira
it.UpdateSubtitle=Mira %1 è installato. Questo programma lo aggiorna alla versione %2.
it.UpdateText=I file del programma vengono sostituiti. La libreria (impostazioni, database, miniature) viene mantenuta e il suo database viene migrato alla nuova versione; prima viene creata una copia di backup.%n%nSe Mira è in esecuzione, viene prima chiuso. Una pipeline in corso viene messa in pausa e riprende al prossimo avvio.
it.UpdateMemoTitle=Aggiornamento:
it.UpdateMemo=Da %1 a %2; la libreria viene mantenuta e il suo database migrato (con una copia di backup)
it.ClosingApp=Chiusura di Mira...
it.MigratingLibrary=Aggiornamento del database della libreria...
it.MigrationFailed=Impossibile aggiornare il database della libreria (errore %1). Mira riproverà al prossimo avvio.%n%nDettagli: logs\migrate.log nella cartella dati della libreria (predefinita: %2).
it.DatabaseTooNew=Il database della libreria proviene da una versione più recente di Mira, che questa versione non può aprire. Reinstalla la versione più recente oppure ripristina una copia di backup del database.

pt.UpdateCaption=Atualizar o Mira: %1 → %2
pt.UpdateTitle=Atualizar o Mira
pt.UpdateSubtitle=O Mira %1 está instalado. Este programa atualiza-o para a versão %2.
pt.UpdateText=Os ficheiros do programa são substituídos. A sua biblioteca (definições, base de dados, miniaturas) é mantida e a respetiva base de dados é migrada para a nova versão; antes é feita uma cópia de segurança.%n%nSe o Mira estiver em execução, é fechado primeiro. Um pipeline em curso é colocado em pausa e continua no próximo arranque.
pt.UpdateMemoTitle=Atualização:
pt.UpdateMemo=De %1 para %2; a biblioteca é mantida e a base de dados migrada (com uma cópia de segurança)
pt.ClosingApp=A fechar o Mira...
pt.MigratingLibrary=A atualizar a base de dados da biblioteca...
pt.MigrationFailed=Não foi possível atualizar a base de dados da biblioteca (erro %1). O Mira tenta novamente no próximo arranque.%n%nDetalhes: logs\migrate.log na pasta de dados da biblioteca (por predefinição %2).
pt.DatabaseTooNew=A base de dados da biblioteca é de uma versão mais recente do Mira, que esta versão não consegue abrir. Volte a instalar a versão mais recente ou restaure uma cópia de segurança da base de dados.

ptbr.UpdateCaption=Atualizar o Mira: %1 → %2
ptbr.UpdateTitle=Atualizar o Mira
ptbr.UpdateSubtitle=O Mira %1 está instalado. Este instalador o atualiza para a versão %2.
ptbr.UpdateText=Os arquivos do programa são substituídos. Sua biblioteca (configurações, banco de dados, miniaturas) é mantida e o banco de dados é migrado para a nova versão; antes é feita uma cópia de segurança.%n%nSe o Mira estiver em execução, ele é fechado primeiro. Um pipeline em andamento é pausado e continua na próxima inicialização.
ptbr.UpdateMemoTitle=Atualização:
ptbr.UpdateMemo=De %1 para %2; a biblioteca é mantida e o banco de dados migrado (com uma cópia de segurança)
ptbr.ClosingApp=Fechando o Mira...
ptbr.MigratingLibrary=Atualizando o banco de dados da biblioteca...
ptbr.MigrationFailed=Não foi possível atualizar o banco de dados da biblioteca (erro %1). O Mira tentará novamente ao iniciar.%n%nDetalhes: logs\migrate.log na pasta de dados da biblioteca (por padrão %2).
ptbr.DatabaseTooNew=O banco de dados da biblioteca é de uma versão mais recente do Mira, que esta versão não consegue abrir. Instale a versão mais recente novamente ou restaure uma cópia de segurança do banco de dados.

nl.UpdateCaption=Mira bijwerken: %1 → %2
nl.UpdateTitle=Mira bijwerken
nl.UpdateSubtitle=Mira %1 is geïnstalleerd. Deze setup werkt het bij naar versie %2.
nl.UpdateText=De programmabestanden worden vervangen. Je bibliotheek (instellingen, database, miniaturen) blijft behouden en de database wordt naar de nieuwe versie gemigreerd; eerst wordt er een reservekopie van gemaakt.%n%nAls Mira actief is, wordt het eerst afgesloten. Een lopende pipeline wordt gepauzeerd en gaat bij de volgende start verder.
nl.UpdateMemoTitle=Bijwerken:
nl.UpdateMemo=Van %1 naar %2; de bibliotheek blijft behouden en de database wordt gemigreerd (met een reservekopie)
nl.ClosingApp=Mira wordt afgesloten...
nl.MigratingLibrary=De database van de bibliotheek wordt bijgewerkt...
nl.MigrationFailed=De database van de bibliotheek kon niet worden bijgewerkt (fout %1). Mira probeert het opnieuw bij de volgende start.%n%nDetails: logs\migrate.log in de gegevensmap van de bibliotheek (standaard %2).
nl.DatabaseTooNew=De database van de bibliotheek komt van een nieuwere versie van Mira, die deze versie niet kan openen. Installeer de nieuwere versie opnieuw of zet een reservekopie van de database terug.

pl.UpdateCaption=Aktualizacja Mira: %1 → %2
pl.UpdateTitle=Aktualizacja Mira
pl.UpdateSubtitle=Mira %1 jest zainstalowana. Ten instalator zaktualizuje ją do wersji %2.
pl.UpdateText=Pliki programu zostaną zastąpione. Twoja biblioteka (ustawienia, baza danych, miniatury) zostanie zachowana, a jej baza danych zostanie przeniesiona do nowej wersji; wcześniej zostanie utworzona kopia zapasowa.%n%nJeśli Mira jest uruchomiona, zostanie najpierw zamknięta. Trwający pipeline zostanie wstrzymany i wznowiony przy następnym uruchomieniu.
pl.UpdateMemoTitle=Aktualizacja:
pl.UpdateMemo=Z wersji %1 do %2; biblioteka zostanie zachowana, a jej baza danych przeniesiona (z kopią zapasową)
pl.ClosingApp=Zamykanie Mira...
pl.MigratingLibrary=Aktualizowanie bazy danych biblioteki...
pl.MigrationFailed=Nie udało się zaktualizować bazy danych biblioteki (błąd %1). Mira spróbuje ponownie przy następnym uruchomieniu.%n%nSzczegóły: logs\migrate.log w folderze danych biblioteki (domyślnie %2).
pl.DatabaseTooNew=Baza danych biblioteki pochodzi z nowszej wersji Mira, której ta wersja nie potrafi otworzyć. Zainstaluj ponownie nowszą wersję lub przywróć kopię zapasową bazy danych.

ru.UpdateCaption=Обновление Mira: %1 → %2
ru.UpdateTitle=Обновление Mira
ru.UpdateSubtitle=Установлена Mira %1. Эта программа обновит её до версии %2.
ru.UpdateText=Файлы программы будут заменены. Ваша библиотека (настройки, база данных, миниатюры) сохраняется, а её база данных переносится на новую версию; перед этим создаётся резервная копия.%n%nЕсли Mira запущена, она сначала будет закрыта. Выполняемый конвейер приостанавливается и продолжится при следующем запуске.
ru.UpdateMemoTitle=Обновление:
ru.UpdateMemo=С %1 на %2; библиотека сохраняется, её база данных переносится (с резервной копией)
ru.ClosingApp=Закрытие Mira...
ru.MigratingLibrary=Обновление базы данных библиотеки...
ru.MigrationFailed=Не удалось обновить базу данных библиотеки (ошибка %1). Mira попробует снова при следующем запуске.%n%nПодробности: logs\migrate.log в папке данных библиотеки (по умолчанию %2).
ru.DatabaseTooNew=База данных библиотеки создана более новой версией Mira, которую эта версия не может открыть. Установите более новую версию снова или восстановите резервную копию базы данных.

ja.UpdateCaption=Mira の更新: %1 → %2
ja.UpdateTitle=Mira の更新
ja.UpdateSubtitle=Mira %1 がインストールされています。このセットアップでバージョン %2 に更新します。
ja.UpdateText=プログラム ファイルは置き換えられます。ライブラリ (設定、データベース、サムネイル) はそのまま残り、データベースは新しいバージョンに移行されます。移行の前にバックアップが作成されます。%n%nMira が実行中の場合は、先に終了します。実行中のパイプラインは一時停止され、次回の起動時に再開します。
ja.UpdateMemoTitle=更新:
ja.UpdateMemo=%1 から %2 へ。ライブラリはそのまま残り、データベースは移行されます (バックアップあり)
ja.ClosingApp=Mira を終了しています...
ja.MigratingLibrary=ライブラリのデータベースを更新しています...
ja.MigrationFailed=ライブラリのデータベースを更新できませんでした (エラー %1)。Mira は次回の起動時に再試行します。%n%n詳細: ライブラリのデータ フォルダー内の logs\migrate.log (既定: %2)
ja.DatabaseTooNew=ライブラリのデータベースは新しいバージョンの Mira で作成されたため、このバージョンでは開けません。新しいバージョンを再インストールするか、データベースのバックアップを復元してください。

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

[Code]
const
  UninstallKey = 'Software\Microsoft\Windows\CurrentVersion\Uninstall\{#emit SetupSetting("AppId")}_is1';
  QuitExitOk = 0;
  MigrateTooNew = 3;

var
  InstalledVersion: String;  { empty: not installed yet }
  UpdatePage: TOutputMsgWizardPage;

{ The version of an installed Mira (from its uninstall entry), or ''. }
function FindInstalledVersion(): String;
begin
  Result := '';
  if not RegQueryStringValue(HKCU, ExpandConstant(UninstallKey), 'DisplayVersion', Result) then
    if not RegQueryStringValue(HKLM, ExpandConstant(UninstallKey), 'DisplayVersion', Result) then
      Result := '';
end;

function InitializeSetup(): Boolean;
begin
  InstalledVersion := FindInstalledVersion();
  Result := True;
end;

procedure InitializeWizard();
begin
  if InstalledVersion = '' then
    Exit;
  WizardForm.Caption := FmtMessage(CustomMessage('UpdateCaption'), [InstalledVersion, '{#AppVersion}']);
  UpdatePage := CreateOutputMsgPage(wpWelcome, CustomMessage('UpdateTitle'),
    FmtMessage(CustomMessage('UpdateSubtitle'), [InstalledVersion, '{#AppVersion}']),
    CustomMessage('UpdateText'));
end;

procedure AddMemo(var Memo: String; const Section, NewLine: String);
begin
  if Section = '' then
    Exit;
  if Memo <> '' then
    Memo := Memo + NewLine + NewLine;
  Memo := Memo + Section;
end;

{ The Ready page: the usual sections, after a note when this is an update. }
function UpdateReadyMemo(Space, NewLine, MemoUserInfoInfo, MemoDirInfo, MemoTypeInfo,
  MemoComponentsInfo, MemoGroupInfo, MemoTasksInfo: String): String;
begin
  Result := '';
  if InstalledVersion <> '' then
    AddMemo(Result, CustomMessage('UpdateMemoTitle') + NewLine + Space
      + FmtMessage(CustomMessage('UpdateMemo'), [InstalledVersion, '{#AppVersion}']), NewLine);
  AddMemo(Result, MemoUserInfoInfo, NewLine);
  AddMemo(Result, MemoDirInfo, NewLine);
  AddMemo(Result, MemoTypeInfo, NewLine);
  AddMemo(Result, MemoComponentsInfo, NewLine);
  AddMemo(Result, MemoGroupInfo, NewLine);
  AddMemo(Result, MemoTasksInfo, NewLine);
end;

{ Runs the command line of the installed (or just installed) app, hidden, as the user who
  started the setup: the library is per user. Returns the exit code, or -1. }
function RunCli(const Params: String): Integer;
var
  Cli: String;
  Code: Integer;
begin
  Result := -1;
  Cli := ExpandConstant('{app}\{#CliExe}');
  if not FileExists(Cli) then
    Exit;
  if ExecAsOriginalUser(Cli, Params, ExpandConstant('{app}'), SW_HIDE, ewWaitUntilTerminated, Code) then
    Result := Code;
  Log(Format('%s %s: exit code %d', [Cli, Params, Result]));
end;

{ Before files are replaced: ask a running Mira to quit as if its window was closed (a running
  pipeline is paused, not killed). The command line of the installed version is used: a newer
  setup's files are not there yet. Versions without quit-app (exit code 2) and apps that do not
  quit leave it to the Restart Manager (CloseApplications). }
function PrepareToInstall(var NeedsRestart: Boolean): String;
var
  Progress: TOutputProgressWizardPage;
  Code: Integer;
begin
  Result := '';
  if not FileExists(ExpandConstant('{app}\{#CliExe}')) then
    Exit;
  Progress := CreateOutputProgressPage(SetupMessage(msgWizardPreparing), CustomMessage('ClosingApp'));
  Progress.SetText(CustomMessage('ClosingApp'), '');
  Progress.Show;
  try
    Code := RunCli('quit-app --timeout 120');
    if Code <> QuitExitOk then
      Log('Mira did not confirm that it quit; the Restart Manager closes it if needed');
  finally
    Progress.Hide;
  end;
end;

{ After the files are installed: migrate the library's database (a backup copy is made first).
  Nothing to do without a library. A failure does not undo the installation: the app migrates
  on start as well. }
procedure MigrateLibrary();
var
  Code: Integer;
begin
  WizardForm.StatusLabel.Caption := CustomMessage('MigratingLibrary');
  WizardForm.FilenameLabel.Caption := '';
  Code := RunCli('migrate --desktop');
  if Code = 0 then
    Exit;
  if Code = MigrateTooNew then
    SuppressibleMsgBox(CustomMessage('DatabaseTooNew'), mbError, MB_OK, IDOK)
  else
    SuppressibleMsgBox(FmtMessage(CustomMessage('MigrationFailed'),
      [IntToStr(Code), ExpandConstant('{localappdata}\vision-tagger\data')]), mbError, MB_OK, IDOK);
end;

procedure CurStepChanged(CurStep: TSetupStep);
begin
  if CurStep = ssPostInstall then
    MigrateLibrary();
end;

{ Uninstalling: close a running Mira gracefully first, so that its files can be removed. The
  library is kept. }
procedure CurUninstallStepChanged(CurUninstallStep: TUninstallStep);
var
  Code: Integer;
begin
  if (CurUninstallStep = usUninstall) and FileExists(ExpandConstant('{app}\{#CliExe}')) then
    Exec(ExpandConstant('{app}\{#CliExe}'), 'quit-app --timeout 60', ExpandConstant('{app}'),
      SW_HIDE, ewWaitUntilTerminated, Code);
end;
