# Sistem za preporuku vesti
## Definicija problema

Cilj projekta je kreiranje sistema za automatsku preporuku vesti, izbegavajući lažne članke, kao i sponzorisane, odnosno reklamne tekstove. Za ovakav sistem, potrebno je proces obrade podataka podeliti u više koraka. Ovaj sistem bi funkcionisao za engleski jezik, zbog postojanja većeg broja javno dostupnih izvora podataka. Prvi deo je uklanjanje nepodesnih članaka, dok bi drugi deo podrazumevao podsistem za preporuku članaka na osnovu preostalog skupa.

## Motivacija

Pretraga za vestima može biti bespotrebno komplikovan proces, sa obzirom na broj članaka koji se objavljuje na različitim izvorima svaki dan. Odličan primer sistema koji rešava ovaj problem je Medium platforma (www.medium.com), koja preporučuje blog članke na osnovu korisnikove istorije korišćenja platforme. Postojanje platforme za preporuku vesti bi moglo biti odlično „one stop“ rešenje za korisnike koji ne žele da listaju velik broj različitih sajtova i aplikacija svaki dan. Negativan aspekt ovakvog sistema bi bio što bi potencijalno mogao preporučivati maliciozne članke, koji bi, u najlakšem slučaju, odvlačili korisnika od korišćenja ovakve aplikacije.


## Skupovi podataka

Za ovakav sistem, biće potrebno koristiti najmanje dva skupa podataka, odnosno po jedan za svaki podsistem.
Jedan od najpopularnijih skupova podataka za filtriranje lažnih vesti je „Fake and real news dataset“, preuzet sa kaggle.com (https://www.kaggle.com/clmentbisaillon/fake-and-real-news-dataset). Ovaj skup sadrži Real i Fake csv datoteke, u kojima se nalaze podaci o naslovu, tekstu, kao i temi članka.
	Za podsistem za preporuku vesti, biće korišten „Mind – Microsoft news recommendation dataset“, preuzet sa microsoft.com (https://www.microsoft.com/en-us/research/publication/mind-a-large-scale-dataset-for-news-recommendation/). Ovaj skup podataka sadrži podatke relevantne za kreiranje profila članaka, kao i podatke za kreiranje profila korisnika. Konkretno, news.tsv fajl sadrži podatke o kategoriji, podkategoriji, naslovu i abstraktu članka, kao i entitete naslova i abstrakta (npr. [{"Label": "PGA Tour", "Type": "O", "WikidataId": "Q910409", "Confidence": 1.0, "OccurrenceOffsets": [0], "SurfaceForms": ["PGA Tour"]}]). Datoteka behaviors.tsv sadrži ID korisnika, vreme, istorijat klikova i impresije u formatu N129416-0 (clanak-kliknuo).

## Metodologija
Kako su maliciozni članci odvojeni od redovnih, potrebno je spojiti dve csv datoteke u jedan skup podataka koji sadrži naznaku da li je vest lažna ili nije (datoteke iz fake.csv su lažne, a iz true.csv nisu). Zbog metodologije koje predlaže „Fake news detection“, najbolje rezultate je moguće postići jedino ukoliko za svaki članak postoji naznačena tema članka. Potrebno je proći kroz skup podataka i proveriti da li svaki članak ima naznačenu temu. Ukoliko nema, postoje tri opcije: ručno naznačiti temu, ukloniti red iz skupa, ili ukoliko previše članaka nemaju temu, pronaći automatsko rešenje za izvlačenje teme, što je najlošija opcija. Nakon čega, potrebno je očistiti tekstove konvertovanjem u mala slova, izbacivanjem stop karaktera i vađenjem brojeva iz teksta. Ovo se sve može postići upotrebom built in python metoda i regexa. Potom se radi tokenizacija teksta, ograničavanjem gornjih granica broja reči i broja karaktera u svakoj sekvenci. Nakon toga, sekvence dobijaju padding, kako bi bile istog oblika. Labele se konvertuju u brojeve 0-1. Nakon ovoga, tekstovi se provlače kroz LSTM, nakon koga se prolazi kroz potpuno povezan sloj koji vrši binarnu klasifikaciju na true ili fake. Poboljšanje rezultata se može dobiti provlačenjem i tema tekstova kroz iste procese, pa spajanjem rezultata na kraju. 
	Za podsistem za preporuku vesti postoje tri različite mogućnosti za implementaciju: upotreba gotovog recommender-a, upotreba NPA – kojeg bismo sami implementirali od nule i upotreba graf neuronske mreže, koju Microsoft predlaže kao potencijalno dobro i inovativno rešenje. Kako bismo ovaj projekat spojili sa Neuronskim mrežama i kako nam je cilj da pokušamo da implementiramo što veći i kompleksniji sistem, ideja je da implementiramo izmenjenu verziju NPA (koji je opisan u radu 3), a da koristimo već postojeće recommender sisteme kao vrednost naspram koje bismo poredili naše rešenje.
	Kako bismo koristili NPA, primenili bismo sličnu metodologiju kao u radu 3. Potrebno je podeliti sistem u 3 dela, enkoder vesti, enkoder korisnika i klik prediktor. Enkoder vesti koristi word2vec kako bi vektorizovao naslove (ili abstrakte – zavisi od implementacije do implementacije). Rezultujući vektori se šalju u CNN, koji izvlači lokalne kontekste između reči – jedna reč sama po sebi ne mora imati specifično značenje, ali kombinacija dve ili tri reči može imati značajan uticaj na sistem preporuke.  Rezultati se iz CNN-a šalju u BERT, kojeg koristimo da kreiramo word level attention vektore, koji analiziraju na kojim rečima u naslovima će biti fokus.
	Za korisnički enkoder postoje dva pristupa – generički „newsrec“, koji manje više svi recommenderi koriste na sličan način i „LSTUR“ (long short term user representation). Prvi pristup podrazumeva primenu enkodera vesti na sve članke koje je korisnik čitao i sumarizaciju rezultata određenim metodama (poput metode spomenute u radu 3). Drugi, složeniji pristup je nama interesantniji, jer kombinuje dugoročna i kratkotrajna interesovanja korisnika. Konkretno, kratkoročna interesovanja korisnika se mogu modelovati kroz analizu nedavno pregledanih tekstova, upotrebom rekurentnih mreža koje mogu izvući obrasce korisnikovih kratkoročnih preferenci. Ovo podrazumeva provlačenje kandidatskih članaka kroz enkoder vesti i slanje rezultata u rekurentnu mrežu. Dugotrajne preference podrazumevaju embedovanje korisničkog identifikatora, pa kombinovanje sa rezultatima rekurentnih mreža na jedan od dva načina: korišćenje korisničkog embedinga kao prvog elementa u lancu rekurentnih rezultata (tzv. INI pristup), ili konkatenacija na kraju lanca vektora rekurentnih modela (tzv CON pristup). Nije navedena jasna prednost jednog pristupa u odnosu na drugi, a u radu 2 navedeno je da su oba pristupa skoro podjednako dobra (0,05% AUC metrike, 0,02%-0,03% u ostalim metrikama prednosti za INI pristup). 
	Kada su u pitanju parametri, za podsistem za preporuku vesti je preporučeno sledeće:
dimenzija embedding vektora teksta je 200 (MIND dokumentacija), broj filtera na CNN je 300, window dimenzija CNN-a je 3. Optimizator je adam sa learning rate-om od 0.01, sa batch size 400. Broj negativnih uzoraka naspram pozitivnih je 4:1 (rad 3). Svi ovi parametri su određeni eksperimentalno nad validacionim skupom podataka.
Parametri za podsistem za detekciju lažnih vesti: maksimalan broj reči je 50.000, maksimalna dužina sekvence je 250 reči, dužina embeddinga je 100 (izvor: https://towardsdatascience.com/multi-class-text-classification-with-lstm-1590bee1bd17). Ostali parametri će biti utvrđeni eksperimentalnim putem.

## Metod evaluacije
Kako ova dva sistema funkcionišu potpuno odvojeno i treniraju se potpuno odvojenim izvorima podataka, odvojili bismo i evaluaciju oba dela sistema. 
Fake news skup je preporučeno podeliti na 80 – 20 razmeru (training - test). Kao meru evaluacije koristićemo preciznost predikcije da li je vest True ili Fake (u odnosu na polje koje ćemo dobiti spajanjem true i fake csv datoteka).
Mind skup je malo kompleksniji, jer ne postoji jedno obeležje koje diktira kolika je verovatnoća da će korisnik kliknuti na neki članak. Rešenje koje rad 3 predlaže se pojavljuje i u dosta drugih radova na temu preporuke vesti, uključujući i članke o LSTUR pristupu i zvaničnu Microsoft dokumentaciju, a to je korišćenje „Negative sampling“ tehnike. Ovo podrazumeva da se za svaki pozitivan uzorak (korisnik kliknuo na vest) uzme K negativnih uzoraka (korisnik nije kliknuo na vest). Ovako se proces predikcije pretvara u klasifikacioni problem sa K + 1 klasom. Formula za ovu tehniku je navedena u opisu rada 3. Kako bismo izračunali AUC i ostale metrike, možemo upotrebiti Microsoft-ov evaluator.py, koji se koristi kao benchmark za ostale prediktore. On sadrži realne verovatnoće da će korisnik kliknuti na neki naslov, koje poredi sa verovatnoćama koje naš sistem može proizvesti i generiše metrike relevantne za naš sistem.

## Literatura
[1]	Maramreddy, Yogi Reddy & Gokul, Prathin & Asamani,. (2021). News Recommendation System. The International journal of analytical and experimental modal analysis. 13. 809.
•	Zadatak rada:
Kreiranje sistema za preporuku, poput sistema za preporuku artikala i sadržaja na društvenim platformama, samo specijalizovanog za vesti i novinske članke.
•	Metodologija:
Podaci su prvobitno scrape-ovani sa različitih portala i sajtova za arhiviranje vesti. Nakon prikupljanja, podaci su prečišćeni tako što su dodate nepostojane vrednosti za bitne kolone poput teme članka. Nakon ovih koraka, primenjene su tokenizacija, lematizacija, stemming, kao i uklanjanje oznaka interpunkcije nad tekstovima članaka. 

•	Validacija:
Mera preciznosti u odnosu na prikupljene podatke.

•	Rezultat:
Nije naveden rezultat kao mera u procentima, nego su predstavljene osnovne tehnike uz pomoć kojih je moguće dobiti dobre rezultate. 

•	Mišljenje:
Rad se ne bavi konkretnim istraživanjima, ali navodi brojne korisne tehnike koje znatno mogu olakšati rad i navodi par bitnih teza za razumevanje načina funkcionisanja postojećih skupova podataka.

•	Zaključak:
Word2vec algoritam, koji ovaj rad predlaže, bismo mogli upotrebiti za vektoritzaciju reči, ukoliko to ne bude moguće uz pomoć CNN-a. Takođe, za pretprocesiranje tekstova samih članaka, moglo bi biti korisno uraditi lematizaciju, kako bi se smanjila njihova veličina, a očuvao kontekst. 

[2]	Wu, Fangzhao & Qiao, Ying & Chen, Jiun-Hung & Wu, Chuhan & Qi, Tao & Lian, Jianxun & Liu, Danyang & Xie, Xing & Gao, Jianfeng & Wu, Winnie & Zhou, Ming. (2020). MIND: A Large-scale Dataset for News Recommendation. 3597-3606. 10.18653/v1/2020.acl-main.331.
•	Zadatak rada: 
Kreiranje i dokumentovanje načina funkcionisanja MIND skupa podataka.

•	Metodologija:
Sakupljeni su podaci korisnika Microsoftove platforme za onlajn plasiranje vesti korisnicima (https://news.microsoft.com/), pa su obradjeni izbacivanjem korisničkih podataka, kao i predviđanjem verovatnoće da će korisnik kliknuti na neki članak uz pomoć više različitih „recommender“ algoritama, koje su open source-ovali.

•	Validacija:
Preciznost pojedinačnih modela je merena korišćenjem AUC (area under curve) i mDCG@10 metrike. Na osnovu tih mera određen je najprecizniji recommender, na osnovu koga su definisani Confidence atributi pojedinačnih naslova, koji će biti relevantni za upotrebu skupa podataka.

•	Rezultat:
Najprecizniji rezultati su postignuti korišćenjem kombinacije LSTM i attention mehanizma – 66,91% AUC mera, kao i 40.85% mDCG@10 metrike.

•	Mišljenje:
Rad direktno daje opis načina funkcionisanja skupa podataka koji će biti korišćen za kreiranje preporuka vesti, pa će biti od velikog značaja za izradu rešenja.

•	Zaključak:
Postoji više načina na koji bismo mogli upotrebiti ovaj rad i vezane članke. Najlakši način bi bio upotreba gotovih, već izrađenih recommender modela, za koje bismo morali pripremiti podatke upotrebom odgovarajućih metoda koje se razlikuju od modela do modela. Zajedničko za većinu modela jeste da bismo morali skratiti tekstove upotrebom lematizacije ili stemminga, pa potom vektorizovati tekst, jer većina modela koristi Bert ili slične modele, koji imaju problem sa ulazima velike dužine.
Alternativni pristup bi bio da sami napravimo recommender, što nam je cilj. Pristup koji bismo probali da rekreiramo je NPA (NPA: Neural News Recommendation with Personalized Attention), o kome će biti reči u sledećem radu.

[3]	Wu, Chuhan & Wu, Fangzhao & An, Mingxiao & Huang, Jianqiang & Huang, Yongfeng & Xie, Xing. (2019). NPA: Neural News Recommendation with Personalized Attention. 10.1145/3292500.3330665.
•	Zadatak rada:
Kreiranje radnog okvira za preporuku vesti na osnovu obrade semantike tekstova koje je korisnik već čitao, kao i na osnovu semantike dostupnih tekstova.

•	Metodologija:
Analiza članaka se svodi na 3 koraka: „enkoder vesti“, „enkoder korisnika“ i „klik prediktor“. Enkoder vesti uzima naslov članka i pretvara ga u sekvencu nisko dimenzionih „dense“ vektora. Nakon ovoga, vektori se šalju u CNN, kako bi se izvukli lokalni konteksti reči. Konkretno, dat je primer naslova „The best Fiesta bowl moments“, gde „Fiesta“ i „bowl“ nemaju relevantno značenje dok se ne uzmu obe reči u obzir, što CNN radi. Izlaz iz CNNa je sekvenca kontekstualnih reprezentacija reči. Ovaj izlaz se dalje šalje u „word-level personalized attention network“. Dok drugi pristupi koriste obične attention mreže, ovaj rad predlaže upotrebu personalizovanih attention mreža, koji uzima u obzir preference korisnika. Ovo se postiže embedovanjem ID-a korisnika u reprezentacioni vektor, nakon čega se koristi dense sloj, koji word-level preference korisnika po formuli qw = ReLU(Vw × eu + vw ), gde su Vw i vw parametri ∈ R, dok je eu embedovan ID korisnika. Ovaj enkoder se primenjuje na sve kandidate članke i na sve pregledane članke.
Korisnički enkoder funkcioniše na sličan način kao i enkoder vesti, konkretno primenjujući attention mrežu sa embedovanim korisničkim ID-em kao u prethodnom koraku. Konačan izlaz iz ovog koraka je suma kontekstualnih reprezentacija naslova pomnožena sa njihovim attention težinama.
Klik prediktor je najjednostavniji deo sistema i predstavlja komponentu koja predviđa verovatnoću da će korisnik kliknuti na određeni naslov. Problematika koja se javlja u ovom koraku je to što je potrebno imati pozitivne primere (šta bi realno korisnik kliknuo) i negativne primere (to što korisnik ne bi kliknuo). Drugi radni okviri predlažu ručno balansiranje pozitivnih i negativnih primera, što na MIND dataset-u sa milion naslova nije jednostavan zadatak. Ovaj rad predlaže negative sampling tehnike, koje kažu da se uzme jedan pozitivan primer za korisnika i K negativnih primera. Način na koji se ovi podaci obrađuju predstavljen je formulama:
y′i = ri′Tu i yi = exp( y′i ) / sum(exp( y′j )), nakon čega se yi normalizuje softmax funkcijom.

•	Validacija:
Kao i u prethodnom radu, validacija je rađena uz pomoć AUC, MRR, mDCG5 i nDCG10 metrika. Nakon kalkulacije ovih rezultata, poređeni su sa drugim kompetetivnim prediktorima vesti poput LibFM, DKN, Dfm i dr.

•	Rezultat:
Postignuti su rezultati – 62,43% AUC mera, kao i 35,35% mDCG@10 metrike. Ovaj rezultat je za nijansu lošiji nego LSTM pristup iz prethodnog rada, ali kada se uporede brzine izvršavanja predikcija i brzine treniranja mreže, ovaj pristup je brži između 20% i 50% u odnosu na ostale konkurentne radne okvire. Takođe, kada se testira dati skup na realnim podacima izvučenim sa sajtova portala, stabilnost ovog pristupa prevladava i čini ga efikasnijim od LSTM-a.

•	Mišljenje:
Ovaj rad daje najjednostavnije od rešenja koje Microsoft predlaže u recommenders paketu, a ujedno je i jedno od preciznijih, pa i realnijih za implementaciju. Takođe, ovo je jedan od retkih radova na datu temu gde su velika većina koraka potpuno transparentno opisana, dok su u drugim radovima često sakrivene optimizacije i međukoraci potrebni da se postigne navedeni rezultat.

•	Zaključak:
Ovaj rad bismo mogli iskoristiti da implementiramo deo sistema koji služi za predikciju naslova uz par izmena. Konkretno, upotrebljene attention mreže sa embeddingom korisnikovog ID-a ne daju znatno bolje rezultate – okvirno ispod 1%, pa bi moglo biti korisnije zameniti upotrebljene attention mreže sa modernijim rešenjima poput Bert modela, koji se mogu upotrebiti u iste svrhe, a verovatno dalju preciznija rešenja. Takođe, click prediktor bi potencijalno mogao biti izmenjen. Konkretno, ručno balansiranje skupa podataka može drastično povećati preciznost, ali je potrebno više vremena. Ovaj pristup je potencijalno nezgodan, jer je moguće izgubiti informacije koje su očuvane u negativnim primerima kojih će uvek biti više od pozitivnih.

[3]	Winkelmann, Sebastian & Yousefi, Shakir & Fabricius-Bjerre, Frederik. (2020). FAKE NEWS DETECTION.
•	Zadatak rada:
Kreiranje mehanizma za označavanje istinitih, odnosno malicioznih vesti.

•	Metodologija:
Prikupljeni su članci sa različitih sajtova za arhiviranje vesti, pa je ručno navedeno da li je članak istinit ili nije, kao i koji je tip članka. Dalja obrada je podrazumevala pripremu članaka za svaki od pojedinačnih modela koji su testirani. Tačni koraci obrade za svaki pojedinačan model nisu navedeni.

•	Validacija:
Mera evaluacije je preciznost klasifikacije članaka na Real / Fake.
•	Rezultat:
Najveća preciznost je postignuta uz pomoć višeslojnog perceptrona sa 2 skrivena sloja, sa ReLU – ReLu – Sigmoid aktivacionim funkcijama, Adam optimizatorom, learning rate od 0.001, 5 epoha i batch size od 256. Sa datom konfiguracijom je postignuta preciznost od 94.3/88.4 (%) na validacionon i testnom skupu. Kada je sistem testiran na LIAR skupu podataka sa Kaggle, linearnom regresijom je postignuta preciznost od 77.8/60.4 (%), koja je veća u odnosu na višeslojni perceptron koji je postigao 72.8/59.4 (%).

•	Mišljenje:
Rad na veoma jednostavan način prikazuje načine podele članaka na istinite i lažne, pa će značajno olakšati proces filtriranja naslova.

•	Zaključak:
Iako ovaj rad predlaže jednostavno rešenje koje sasvim dobro obavlja posao, potencijalno bi se moglo unaprediti korišćenjem LSTM-a, koji veoma precizno može klasifikovati tekstove, pogotovo jer postoje samo dve klase. Pristup koji je moguće koristiti za sam korak klasifikacije je dat na linku: https://towardsdatascience.com/multi-class-text-classification-with-lstm-1590bee1bd17.
