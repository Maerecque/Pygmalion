pyinstaller --noconfirm --onefile --console --icon "Source/support_files/logo.ico" --name "Pygmalion" --clean --add-data "Source;./Source" --add-data "presets.ini;." "main.py"
move output\Pygmalion.exe .
echo copied