# Importieren der notwendigen Bibliotheken
import pandas as pd  # Datenanalyse und -manipulation
import seaborn as sns  # Statistische Datenvisualisierung
from matplotlib import pyplot as plt  # Grafiken und Diagramme
from sklearn.cluster import KMeans  # K-Means-Clusteralgorithmus
from yellowbrick.cluster import KElbowVisualizer  # Elbogenmethode für die optimale Anzahl von Clustern
from sklearn.preprocessing import StandardScaler  # Standardisierung von Daten

# 1.1 Laden der Daten aus einer CSV-Datei
path = "parkinsons_updrs.data"  # Pfad zur CSV-Datei
data = pd.read_csv(path, delimiter=",")  # Lesen der CSV-Datei

# 1.2 Anzeigen der ersten und letzten fünf Zeilen der Daten
print('#'*100)  # Trenner für die Ausgabe
print('Aufgabe 1.2')  # Beschriftung der Ausgabe
print(data.head())  # Ausgabe der ersten fünf Zeilen
print(data.tail())  # Ausgabe der letzten fünf Zeilen

# 1.3 Ausgabe von Zeilen- und Spaltenanzahl sowie Datentypen
print('#'*100)
print('Aufgabe 1.3')
print(data.info())  # Anzeige von Spaltennamen, Datentypen und Anzahl der Nicht-Null-Werte

# 2.1 Überprüfen auf fehlende Werte und entsprechende Behandlung
print('#'*100)
print('Aufgabe 2.1')
print(data.isnull().any())  # Gibt an, ob es fehlende Werte in den Spalten gibt

# 2.2 Duplikate entfernen
print('#'*100)
print('Aufgabe 2.2')
print(data.duplicated())  # Überprüfung auf Duplikate in den Daten
data = data.drop_duplicates()  # Entfernen von Duplikaten
print(data.duplicated())  # Bestätigung, dass die Duplikate entfernt wurden

# Bestimmen der optimalen Anzahl von Clustern mit der Elbogenmethode
model = KMeans()  # K-Means-Modell ohne spezifizierte Clusteranzahl
visualizer = KElbowVisualizer(model, K=(1, 15), timings=False)  # Elbogenvisualisierung für 1 bis 15 Cluster
visualizer.fit(data)  # Anpassen des Modells an die Daten
visualizer.show()  # Anzeigen der Elbogenvisualisierung, um die optimale Anzahl von Clustern zu finden

# K-Means-Clustering mit 4 Clustern
KMeans = KMeans(n_clusters=4)  # Clusteranzahl auf 4 festgelegt
pred = KMeans.fit_predict(data)  # Anpassen des Modells und Vorhersagen der Cluster
data_new = pd.concat([data, pd.DataFrame(pred, columns=["label"])], axis=1)  # Hinzufügen der Clusterlabels zu den Daten
print(data_new)  # Ausgabe der Daten mit den Clusterlabels

# 3.1 Statistische Kennzahlen berechnen
print('#'*100)
print('Aufgabe 3.1')
descriptive_statistic = data.describe()  # Berechnung grundlegender Statistik
print(descriptive_statistic)  # Ausgabe der Statistik

# 3.2 Korrelation zwischen Eigenschaften ermitteln
print('#'*100)
print('Aufgabe 3.2')
correlation = data.corr()  # Korrelation zwischen allen Spalten
print(correlation['age'].sort_values())  # Ausgabe der Korrelation von 'age' zu anderen Eigenschaften, sortiert nach Wert

# 4.1 Histogramm erstellen
print('#'*100)
print('Aufgabe 4.1')
print('Histogramm')
data['sex'].hist()  # Histogramm der 'sex'-Spalte
plt.title('Verteilung des Geschlechts')  # Titel für das Diagramm
plt.show()  # Anzeigen des Histogramms

# 4.2 Scatterplot erstellen
print('#'*100)
print('Aufgabe 4.2')
print('Scatterplot')
plt.scatter(data['age'], data['test_time'])  # Scatterplot für 'age' vs. 'test_time'
plt.title('Alter vs. Testzeit')  # Titel für den Scatterplot
plt.show()  # Anzeigen des Scatterplots

# 4.3 Boxplot erstellen
print('#'*100)
print('Aufgabe 4.3')
print('Boxplot')
data.boxplot(column='RPDE', by='DFA')  # Boxplot von 'RPDE' gruppiert nach 'DFA'
plt.title('Boxplot von RPDE nach DFA')  # Titel für den Boxplot
plt.show()  # Anzeigen des Boxplots
