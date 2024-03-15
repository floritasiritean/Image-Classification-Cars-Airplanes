# Image-Classification-Cars-Airplanes
Antrenarea unui algoritm de clasificare(KNN), utilizand un set de date.

1. Descrierea Proiectului
Proiectul își propune să implementeze un clasificator pentru a diferenția imaginile care prezintă mașini de imaginile care prezintă avioane folosind algoritmul de învățare supervizată k-NN. Acesta este un proiect de învățare automată simplu, ce se concentrează pe utilizarea momentelor Hu ca descriptori de formă pentru extragerea caracteristicilor imaginilor.

2. Setul de Date
Setul de date este compus din două categorii principale: imagini cu mașini și imagini cu avioane, având extensia .jpg. Aceste imagini sunt organizate în două directoare distincte, "Masini" și
"Avioane". Imaginile au fost preprocesate prin redimensionare la o dimensiune standard de 64x64 pixeli. Etichetele asociate fiecărei imagini sunt atribuite în funcție de numele fișierului: imagini care
încep cu "m" sunt etichetate drept mașini (clasa 1), iar celelalte sunt etichetate drept avioane (clasa0).

4. Algoritmul Utilizat
Pentru clasificare, am ales algoritmul k-NN (k vecini cei mai apropiați), implementat prin intermediul bibliotecii Scikit-learn în Python. Setul de date a fost împărțit într-un set de antrenare (80%) și un set de validare (20%) folosind funcția train_test_split. Clasificatorul a fost antrenat pe setul de antrenare și evaluat pe setul de validare pentru ajustarea parametrilor.

5. Implementarea Codului
Proiectul este implementat în Python și utilizează biblioteci precum: OpenCV pentru prelucrarea imaginilor, NumPy pentru manipularea datelor, Scikit-learn pentru implementarea k-NN și metrici de evaluare, precum și Matplotlib și Seaborn pentru vizualizare.
