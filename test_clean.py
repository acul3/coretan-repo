import pytest

from c4nlpreproc.clean.clean import clean_text
from c4nlpreproc.clean.clean import count_dict

def test_clean_text_tweet():
  text = """@Pierre Le Blanc#212950 Net het boek gelezen Heimwee naar Hairos,daar kwam je naam ook regelmatig in voor,mooie boek trouwens.
piettolsma Ja, in Den-Haag zeggen we dan altijd " Ze kunnen beter over je fiets lullen dan over je l.. fietsen "
"""
  expected = None
  assert clean_text(text) == expected

def test_clean_text_gamed():

  text = """Gamed.nl - Korte film, trailer en datum van Watch Dogs Legion
Door Rene Groen op 12-07-2020 om 21:14
Watch Dogs: Legion had bij de Ubisoft Forward presentatie de eer om af te trappen. Ze deden dit middels een korte film en een meer uitgebreide presentatie van de gameplay waarin we zien dat missies op verschillende manieren aangepakt kunnen worden, afhankelijk van het karakter dat je kiest. We leren ook dat de game vanaf 29 oktober verkrijgbaar is.
Andere berichten over Watch Dogs: Legion [05-08-2020] Bekijk de Whistleblower missie van Watch Dogs: Legion [23-07-2020] Watch Dogs: Legion met Resistance trailer [23-04-2020] Watch Dogs Legion is launchgame op nieuwe consoles? [19-08-2019] [GC] Verwekom de resistentie uit Watch Dogs [19-08-2019] [GC] Watch Dogs: Legion met RTX beelden [29-07-2019] Watch Dogs Legion progressie uitgelegd [24-07-2019] Watch Dogs Legion heeft vijf verhaallijnen [24-06-2019] Zie het Londen van Watch Dogs: Legion [14-06-2019] [E3] Bedien de stad in Watch Dogs: Legion [13-06-2019] [E3] Gameplaydemo Watch Dogs: Legion [10-06-2019] [E3] Watch Dogs: Legion toont zijn NPC's
21:33 Assassin's Creed Valhalla verschijnt 17 november
14:03 void TRRLM(); //Void Terrarium"""
  expected = None

  result = clean_text(text)
  assert result == expected

def test_clean_text_informatiebijeenkomst():

  text = """Informatiebijeenkomst "Leven met de gevolgen van kanker" Vermoeidheid en energieverdeling | Agenda | Oncologiecentrum
Informatiebijeenkomst "Leven met de gevolgen van kanker" Vermoeidheid en energieverdeling
Home › Actueel › Agenda › Informatiebijeenkomst "Leven met de gevolgen van kanker" Vermoeidheid en energieverdeling
Vermoeidheid tijdens en na de behandeling van kanker komt bij veel mensen voor. Deze vermoeidheid is vaak onvoorspelbaar en kent verschillende vormen. Het kan heel fysiek in het hele lichaam aanwezig zijn maar kan ook mentaal van aard zijn: te moe om veel prikkels aan te kunnen of bijvoorbeeld om snel van onderwerp te wisselen. In het dagelijks leven zijn beide vormen soms moeilijk te onderscheiden. Ze kunnen belemmerend zijn voor het uitvoeren van de dagelijkse activiteiten. In deze bijeenkomst staan we stil bij het ontstaan van en het herkennen van verschillende soorten vermoeidheid. Tevens geven we uitleg over factoren die het moe zijn in stand houden zoals verstoord slaap/waakritme, onvoldoende ervaren van sociale steun, niet-helpende gedachten, angst voor terugkeer, onvoldoende verwerking van de ziekte en problemen in activiteiten regulatie zoals het hebben van een balans in activiteiten gedurende de dag. Maar ook weten welke activiteiten energie kosten en welke energie geven. We reiken daarbij adviezen en handvatten aan hoe om te gaan met de vermoeidheidsklachten."""

  expected = """Vermoeidheid tijdens en na de behandeling van kanker komt bij veel mensen voor. Deze vermoeidheid is vaak onvoorspelbaar en kent verschillende vormen. Het kan heel fysiek in het hele lichaam aanwezig zijn maar kan ook mentaal van aard zijn: te moe om veel prikkels aan te kunnen of bijvoorbeeld om snel van onderwerp te wisselen. In het dagelijks leven zijn beide vormen soms moeilijk te onderscheiden. Ze kunnen belemmerend zijn voor het uitvoeren van de dagelijkse activiteiten. In deze bijeenkomst staan we stil bij het ontstaan van en het herkennen van verschillende soorten vermoeidheid. Tevens geven we uitleg over factoren die het moe zijn in stand houden zoals verstoord slaap/waakritme, onvoldoende ervaren van sociale steun, niet-helpende gedachten, angst voor terugkeer, onvoldoende verwerking van de ziekte en problemen in activiteiten regulatie zoals het hebben van een balans in activiteiten gedurende de dag. Maar ook weten welke activiteiten energie kosten en welke energie geven. We reiken daarbij adviezen en handvatten aan hoe om te gaan met de vermoeidheidsklachten."""

  assert clean_text(text) == expected

def test_clean_text_recept():
  text = """Kwark exotic – Puur Gezond
Kwark exotic
Op: 22 mei 2017
Een elegant dessert, boordevol verschillende smaken en texturen, maar toch makkelijk te bereiden. Licht, lief voor de lijn en bovenal… lekker.
een hand vol pistachenoot gepeld en ongezouten
een handvol kokosschilfer
½ Mango in kleine blokjes
¼ limoen sap en zeste
160 g platte kaas volle
1/2 passievrucht de pulp van
½ tl kokosbloesemsuiker
Schep de mangoblokjes om met ½ tl limoensap (of voeg wat meer toe naar je eigen smaak) en de kokosrasp. Zet opzij in de koelkast.
Rooster de pistachenoten in een droog pannetje en hak ze fijn (zonder er meel van te maken). Rooster vervolgens ook de kokosschilfers tot ze net bruin kleuren.
Roer de passievruchtpulp en de kokosbloesemsuiker door de platte kaas.
Verdeel de pistachenootjes over twee glaasjes, schep er de passievruchtkwark en vervolgens de mangosalade bovenop. Werk af met de geroosterde kokosschilfers en rasp er tot slot wat limoenzeste overheen.
Recept door http://www.karolaskitchen.be"""
  expected = """Een elegant dessert, boordevol verschillende smaken en texturen, maar toch makkelijk te bereiden. Licht, lief voor de lijn en bovenal… lekker.
Schep de mangoblokjes om met ½ tl limoensap (of voeg wat meer toe naar je eigen smaak) en de kokosrasp. Zet opzij in de koelkast.
Rooster de pistachenoten in een droog pannetje en hak ze fijn (zonder er meel van te maken). Rooster vervolgens ook de kokosschilfers tot ze net bruin kleuren.
Roer de passievruchtpulp en de kokosbloesemsuiker door de platte kaas.
Verdeel de pistachenootjes over twee glaasjes, schep er de passievruchtkwark en vervolgens de mangosalade bovenop. Werk af met de geroosterde kokosschilfers en rasp er tot slot wat limoenzeste overheen."""

  result = clean_text(text)
  assert result == expected

def test_clean_text_autoadd():

  text = """Kia Sportage 1.6 ExecutiveLine 12-2016 NL-auto ! - AutoWeek.nl
Kia Sportage 1.6 ExecutiveLine 12-2016 NL-auto !
KM stand 28.996
Kenteken KX-444-P Check kenteken
Nette zuivere NL-dealerauto ! NW door ons geleverd en onderhouden.
Ik wil graag een proefrit maken met Kia Sportage met kenteken KX-444-P...
Ik heb een vraag over de Kia Sportage met kenteken KX-444-P...
Ik wil graag mijn auto inruilen voor de Kia Sportage met kenteken KX-444-P...
KX-444-P"""
  expected = None

  result = clean_text(text)
  assert result == expected


def test_clean_text_vacature():
  text = """Ton de de Jager | Utrecht - At Monday
Ton de de Jager
Ik ben een ervaren accountmanager die zowel in het bedrijfsleven als binnen het Hoger Onderwijs heeft gewerkt, in binnen- en buitenland. De focus lag hierbij op het leveren van diensten op het vlak…
Ik ben een ervaren accountmanager die zowel in het bedrijfsleven als binnen het Hoger Onderwijs heeft gewerkt, in binnen- en buitenland. De focus lag hierbij op het leveren van diensten op het vlak van Learning&Development, Enterprise Architectuur en BPM.
Mijn aanpak wordt gekenmerkt door een grote focus op het resultaat en betrokkenheid bij de mensen waar ik mee samenwerk (klanten en collega’s), door een planmatige werkwijze en het vermogen om in te spelen op veranderingen. Ik kan goed relaties ontwikkelen met diverse partijen in mijn werkomgeving, vooral door vertrouwen te wekken en oprechte interesse te hebben. Afspraak is afspraak, plezier in het werk en enthousiasme zijn voor mij belangrijke waarden.
Ik werk graag in een dynamische omgeving, voor en met professionals uit verschillende disciplines en nivo’s. Het ontdekken van en omgaan met cultuurverschillen zie ik als een verrijking.
Hogeschool van Arnhem en Nijmegen (2016)  Docent Algemene Economie bij het de Faculteit Economie en Management, stagebegeleider Het verzorgen van colleges voor studenten Bedrijfseconomie, Accountancy, Financial Services en Fiscaal Recht en Economie
ValueBlue (2015)  Sales Executive voor alle klanten in Midden- en Zuid Nederland Eindverantwoordelijk voor alle business development en sales in de zuidelijke helft van Nederland. Wij leveren diensten op het vlak van Enterprise Architecture en Beheer Innovatie, en leveren een tool om het ICT-landschap mee in beeld te brengen. Speerpuntmarkten: Gezondheidszorg (Cure en Care) en Gemeenten
BiZZdesign (2013-2015)  Account Manager voor de private-markt Business development en sales op het terrein van Business Process Management, LEAN, Business Modelling en Enterprise Architecture, gericht op tooling, consultancy en opleidingsdiensten.
Capgemini Academy (2002-2012)  Account Manager voor opleidingsdiensten en consultancy in diverse marktsectoren Business development en sales bij top500-bedrijven en overheid in NL en internationaal; verantwoordelijk voor het commerciele resultaat binnen verschillende marktsectoren; eindverantwoordelijk voor het label Architecture Training Services
Capgemini (1999-2002)  Service Manager Overall verantwoordelijk voor de Kwaliteit en P/L van een aantal grote servicecontracten op het vlak van Information Systems Management.
Universiteit Utrecht, Academisch Computer Centrum (1984-1999)  Docent programmeertalen  Afdelingsmanager Educatie en Gebruikersondersteuning  Lid Management Team
Universiteit Utrecht, subfaculteit Aardwetenschappen (1983-1984)  Wetenschappelijk medewerker, docent
Gymnasium B  Kandidaatsexamen Geologie  Doctoraalexamen Geofysica van de vaste aarde
 Leergang Marketing van Diensten  Leergang Financieel Management  Leergang Mobiliteit  Prince2 Foundation
 REED Persoonlijke Effectiviteit  Workshop Facilitation
en veel trainingen op het vlak van methoden en technieken in de ICT.
Wielrennen, Fotografie, Reizen
Sector: Handel (detail), Onderwijs
Klik om de werkgebieden van Ton de de Jager te bekijken
CV van Ton de de Jager
Bekijk het volledige profiel van Ton de de Jager"""
  expected = """Ik ben een ervaren accountmanager die zowel in het bedrijfsleven als binnen het Hoger Onderwijs heeft gewerkt, in binnen- en buitenland. De focus lag hierbij op het leveren van diensten op het vlak van Learning&Development, Enterprise Architectuur en BPM.
Mijn aanpak wordt gekenmerkt door een grote focus op het resultaat en betrokkenheid bij de mensen waar ik mee samenwerk (klanten en collega’s), door een planmatige werkwijze en het vermogen om in te spelen op veranderingen. Ik kan goed relaties ontwikkelen met diverse partijen in mijn werkomgeving, vooral door vertrouwen te wekken en oprechte interesse te hebben. Afspraak is afspraak, plezier in het werk en enthousiasme zijn voor mij belangrijke waarden.
Ik werk graag in een dynamische omgeving, voor en met professionals uit verschillende disciplines en nivo’s. Het ontdekken van en omgaan met cultuurverschillen zie ik als een verrijking.
BiZZdesign (2013-2015)  Account Manager voor de private-markt Business development en sales op het terrein van Business Process Management, LEAN, Business Modelling en Enterprise Architecture, gericht op tooling, consultancy en opleidingsdiensten.
Capgemini (1999-2002)  Service Manager Overall verantwoordelijk voor de Kwaliteit en P/L van een aantal grote servicecontracten op het vlak van Information Systems Management.
en veel trainingen op het vlak van methoden en technieken in de ICT."""

  result = clean_text(text)
  print("asdl",result)
  assert result == expected


def test_clean_text_bagger():
  text = """glen dale wv zip code zip code for ft sam houston san antonio tx zip codes wilmington ma rison arkansas zip code grand blanc mi zip hauppage zip code espanola nm zip zip code for summerville ga zip code siloam springs ar philipsburg pa zip stoughton, ma zip zip code manti utah zip code 06071 zip code for ozark missouri hoquiam zip pigeon mi zip code 200 frontage rd boston ballard wa zip code zip code for madras oregon danielsville ga zip jewell iowa zip code fox lake wi zip code"""
  expected = None

  result = clean_text(text)
  assert result == expected

def test_clean_text_badwords():
  text = """Sexcontact Almere Prive Sex Maastricht / Sex Voor 20 Euro
By Brian Weich 13.01.2018 13.01.2018
Gratis sexsite ; Website: De massage kan gewoon bij je thuis gebeuren reis…. Kom zelf een seks date maken. Prive ontvangst friesland vind je op mr-glove. Erotische massage oost vlaanderen sex massage almere Gratis online sex nl sexcontact groningen Gratis film sex thaise erotische massage den haag Sex.
Leave a comment Avbryt svar E-postadressen publiceras inte. Herken jij dit ook? Prive Ontvangst Prive Ontvangst Priveontvangst. Webcamsex Webcamsex Webcamdames Camsex ouderensex. Webcamsex Stellen Webcamstel Bottomboy. Over de ouderensex pagina Op deze startpagina vind je de beste website op het gebied van Ouderen sex. Over de beheerder Van Ouderen, voor Ouderen.
Laat mijn strakke kutje, mondje en kontje niet wachten schatjee Lees meer. Jij kom mij oppikk Sexy Getinte dame lekkere geile date met me in brasschaat Maya new stoute studente Marokkaanse knappe meid met dikke billen x. Op zoek naar een lieve serieuze suikeroom!! Kom dan gauw langs Mag ik jouw geile pijp sletje zijn.?
Sexjobs heeft het grootste aanbod dames van plezier die aan thuisontvangst en privéontvangst doen. Neem contact met ze op en je hebt vandaag nog een sexafspraak!. Leave a Reply Cancel reply Your email address will not be published. Free Tube Porno Sxe. U kan ons met of zonder afspraak bezoeken. U bent van harte welkom. Voor informatie, of het maken van een afspraak kan u contact met ons opnemen via ons telefoonnummer.
Zoekresultaten voor AA Almere, Flevoland. Wij hebben dames vanaf 21 jaar tot eind 40 in ons bestand staan die wij snel naar uw privé locatie kunnen sturen. Neem eens een kijkje op onze website voor Vanuit een erotische massage breng ik je in de stemming fantasieën werkelijkheid te maken.
Erotische massages flevoland vind je op mr-glove. Thaise erotische massage almere prive ontvangts - Bollnäs Massage Wij hebben dames vanaf 21 jaar tot eind 40 in ons bestand staan die wij snel naar uw privé locatie kunnen sturen.
Gratis sexsite ; Website: De massage kan gewoon bij je thuis gebeuren reis….
Zelf een Themapagina beheren Inloggen beheerders Startpagina. Sexflims zevenaar oma gratis sex maastricht Reeuwijk sexdate. Hotbianca - ontvang prive een paaldans of lapdance sex in maastricht in het hartje Sorry dat ik geen Amsterdam Alkmaar Almelo Almere Alphen aan den Rijn. Geil chat prive sex maastricht - gratis geile Wil jij een geile date met mij kom mij dan opzoeken ontvang prive netjes en.
Mis jij die aandacht, die arm om je heen, die knuffel waar je zo naar verlangt? Grootste aanbod van sexy dames in Maastricht die aan prive ontvangst doen. Leuke studente van 22 Hallo leuke mannen, Mijn naam is Sophie ik ben een mooie meid van 22 jaar.
Er is te vinden hoe je deze vrouwen kunt bereiken voor bijvoorbeeld dating, webcammen, gratis sex chatten en een. Gratis Porno Filmpjes gratis erotische film carpediem massage sexcams. Heb je zin in een geile date m De dames zijn vooral amateurs die het fijn vindne om seks te hebben voor een leuke vergoeding. Probeer goed uit te vinden waar de locatie is en hoe je er kan komen. Esther is een Nederlandse Beauty."""
  expected = None

  result = clean_text(text)
  assert result == expected

  text = """OmaTruus - Live Camsex met Nederlandse Webcamdames
Meer foto's van OmaTruus
Mijn berichten met OmaTruus
rudolf2016-05-31 17:00:32
ALweer groeibevorderend materiaal.
Zoals gewoonlijk zijn het weerheerlijke foto\\\'s.
OmaTruus2017-12-17 17:00:57
Oma is geil en wil masturberen voor JOU, kom je kijken ?
Heerlijke foto\\\'s weer
OmaTruus2016-07-12 00:01:16
Oma is geil en wil masturberen voor jou, kom je mee kijken
OmaTruus2016-05-31 17:00:32
Oma wil masturberen ze is erg geil, kom jij kijken hoe ze dat doet ?
Over mijHoi ik ben een rijpe Oma van 65 en nog steeds geil, kom je kijken hoe ik masturbeer voor je ? Ben erg kinky hou van plassex en meer...Wil jij eens een kinky oma zien log dan in..
Wanneer ben ik online?Als ik kan ben ik online !!
Wat maakt mij geil?Ik hou ervan dingen of vreemde voorwerpen in mijn kutje te stoppen. Plassex
Waar knap ik op af?Op verzoek in prive heel veel
GrotetietenWebcamdamesCamsexWebcamsexNaaktevrouwenDikketietenSexchatRijpevrouwenMilfsSexcamOmasexSexdatingWeb-cam-sexSexdateGeilemeiden
Reviews (1) over OmaTruus
Schrijf een review over OmaTruus
WildJasmine●
AlexaSQuirt●
SQUIRT2much●"""
  expected = None

  result = clean_text(text)
  assert result == expected

  import pprint
  pprint.pprint(count_dict)


test_clean_text_gamed()