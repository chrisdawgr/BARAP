Mapperne indeholder de forskellige metoder, beskrevet i tilhørende dokument.
SURF/SIFT helperfunctions.py, indeholder forskellige hjælpefunktioner,
der bliver brugt, til blandet illustrationer og grafer.
sifttestsuite.py og surftestsuite.py indeholder kode, der kalder de andre 
.py filer, og udregner korrespondancer - Resultaterne og delresultater bliver 
gemt som .npy (Numpy) filer. Grundet parallelt arbejde med SIFT/SURF/Harris/Moravec,
skal Harris/Surf/Moravec .npy filerne for korrespondancer, indsættes i SIFT mappen,
og funktionen "print_diffs" bruges til at udregne repatibilitymeasure, og skabe
illustrationer.

Koden er rodet og der er meget af den. Det vil kræve en del ekstra arbejde, 
hvis koden skulle kunne bruges som et bibliotek, til at finde korrespondancer.