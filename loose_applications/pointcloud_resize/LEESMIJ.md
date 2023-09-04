# Point Cloud Resizer™
Point Cloud Resizer™ is een Python-script waarmee je puntenwolken in het LAS- of LAZ-formaat kunt openen, aanpassen en opslaan. Je kunt de puntenwolk downsamplen (verkleinen) door een voxelgrootte op te geven en vervolgens het resultaat bekijken of opslaan in een nieuw LAS-bestand.

 Dit script maakt gebruik van de volgende Python bibliotheken:

 - `tkinter` voor de GUI (graphical user interface).
- `laspy` voor het lezen en schrijven van LAS/LAZ-bestanden.
- `open3d` voor het werken met puntenwolken en visualisatie.
- `numpy` voor het bewerken van numerieke gegevens.
- `os` voor het werken met bestandspaden en bestandsbeheer.
- `subprocess` voor het starten van een apart proces voor visualisatie.
- `threading` voor het uitvoeren van bewerkingen in aparte threads om de GUI-reactiviteit te behouden.

 ## Installatie
1. Voor het installeren en gebruik van deze applicatie zijn de volgende applicaties nodig:
   - Command-Prompt voor CAD-gebruikers. Deze kan gevonden worden wanneer het Windows menu geopend worden in de map `_Beheer`.
   - ERDAS IMAGINE met Python 3.7 installatie

2. Start Command-Prompt voor CAD-gebruikers op. Zie vorige stap voor locatie.

3. Navigeer naar de schijf waar applicatie bestanden op staan. Bij het opstarten van Command-Prompt voor CAD-gebruikers is `C:/Windows/System32>` te lezen op de onderste regel. <br/>
   **Als de bestanden op de C:/ schijf staan kan deze stap overgeslagen worden.**   <br/>
   Om te wisselen naar de correcte schijf, typ de desbetreffende schijf: bijvoorbeeld als de bestanden op de `J:/` staan typ dan `J:` en druk op de Enter-toets. Nu zal de nieuwe regel verandert zijn in `J:\>`

4. Navigeer naar de correcte folder voor de code. Als je in de windows verkenner de folder opent zie een locatie balk bovenin staan onder de knoppen `Bestand`, `Start`, `Delen`, `Beeld`. Als op deze balk geklikt word zal de folder locatie blauw geselecteerd worden, kopieer deze. Nadat de folder locatie gekopieerd is ga je terug naar Command-Prompt voor CAD-gebruikers en typ je in `cd <naam van folder locatie>`. <br/> **Voorbeeld: als je folder locatie zich bevindt in `X:\bestandsfolder\` dan typ je eerst `X:` in Command-Prompt voor CAD-gebruikers als je nog niet op de correcte schijf was. Hierna typ je in `cd X:\bestandsfolder\` en klik je op de Enter-toets.**

5. Nu gaan we de Python bibliotheken installeren. Om de Python bibliotheken te installeren typen we in Command-Prompt voor CAD-gebruikers het volgende: `python -m pip install -r requirements.txt`. Als het goed zal in Command-Prompt voor CAD-gebruikers een heleboel tekst verschijnen, zolang geen rode tekst verschijnt gaat alles goed. Gele tekst kan genegeerd worden. Mocht er toch rode tekst verschijnen, kijk goed wat voor een error beschreven staat. Als er staat `ERROR: Could not open requirements file: [Errno 2] No such file or directory: 'requirements.txt'`, dan bevind je je in de verkeerde folder. Kijk dan even goed of tekst die voor het `>` icoontje staat overeenkomt met de folder waar je applicatie opgeslagen staat. Voor andere error berichten moet mogelijk Google geraadpleegd worden. <br/>
**Het is belangrijk om op te merken dat de locatie van dit bestand niet dezelfde plek is als waar het installatie bestand staat, om vanaf deze folder te navigeren naar de correcte locatie typ dan het volgende in Command-Prompt voor CAD-gebruikers: `cd ../..` en herhaal het command aan het begin van deze stap.**

## Gebruik
1. Open de folder waar het bestand `pointCloudResizer.py` staat. In de Windows verkenner selecteer locatie balk, op dezelfde manier als in stap 4 van installatie. In plaats dat je de locatie kopieert druk je op de Backspace-knop (←) op het toetsenbord wanneer de locatie blauw is geselecteerd en typ je in `cmd` en druk je op de Enter-toets. Nu zal Command-Prompt voor CAD-gebruikers starten op de correcte locatie. Om de applicatie te starten typ je in `Python pointCloudResizer.py`. Er zal nu een venster openen met een aantal knoppen.
2. Klik op de knop "Open file" om een LAS- of LAZ-bestand te selecteren.
3. Selecteer een voxelgrootte (de afstand tussen punten) in het "Voxel size" veld.
4. Klik op de knop "Apply subsampling" om de puntenwolk te downsamplen op basis van de opgegeven voxelgrootte.
5. De puntenwolkgrootte na het downsamplen wordt weergegeven in het "Point cloud size after subsampling" veld, en de subsamplingfactor wordt weergegeven in het "Subsampling factor" veld.
6. Je kunt de downgesamplede puntenwolk voorbeeld bekijken door op de knop "Preview" te klikken. Dit opent een 3D-visualisatievenster met de puntenwolk.
7. Als je tevreden bent met het resultaat, kun je de puntenwolk opslaan in een nieuw LAS-bestand door op de knop "Save" te klikken. Hiermee wordt een dialoogvenster geopend waarin je de bestandsnaam en locatie kunt kiezen.
8. Het nieuwe LAS-bestand wordt opgeslagen en een melding verschijnt met de opslaglocatie.

## Opmerkingen
- Dit script is ontworpen voor het werken met LAS- en LAZ-bestanden, veelgebruikte formaten voor puntenwolken in de geomatica en 3D-scanningindustrie.
- Zorg ervoor dat je het `logo.ico`-bestand hebt in dezelfde map als het script voor het weergeven van het pictogram in het venster van de toepassing.
- Als het LAS/LAZ-bestand een onbekend formaat heeft of niet kan worden verwerkt, wordt een foutmelding weergegeven.
- De puntenwolk wordt downgesampled met behulp van een regulier grid op basis van de opgegeven voxelgrootte. Het script normaliseert de grootte van de puntenwolk op basis van de oorspronkelijke dichtheid van punten.

Dit script kan nuttig zijn voor professionals in geomatica, geografische informatiesystemen (GIS) en andere vakgebieden die werken met puntenwolken en georuimtelijke gegevens. Het biedt een eenvoudige manier om puntenwolken te verkleinen voor verdere analyse en visualisatie.