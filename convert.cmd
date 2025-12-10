pyinstaller --noconfirm --onefile --windowed --icon "Source/support_files/logo.ico" --name "Pygmalion" --version-file "Source/support_files/version_info.txt" --clean --add-data "presets.ini;." "main.py"
move dist\Pygmalion.exe .

if exist build rmdir /s /q build
if exist dist rmdir /s /q dist
if exist Pygmalion.spec del /q Pygmalion.spec