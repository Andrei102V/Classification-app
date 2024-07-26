import threading
import tensorflow as tf
from tensorflow.keras.preprocessing import image 
from tensorflow.keras.applications.efficientnet import preprocess_input 
import numpy as np
import pandas as pd
import os
import tkinter as tk
from tkinter import filedialog
from tkinter import ttk
from tkinter import font as tkfont
from PIL import Image, ImageTk
import pygame

pygame.mixer.init()

current_audio_file = None
is_playing = False

csv_file_path = '../Desktop/12/birds_corrected.csv'
data = pd.read_csv(csv_file_path)
model_path = '../Desktop/12/EfficientNet-B0.weights.h5'

# Încărcarea modelului de clasificare
def load_and_test_model(model_path, test_image_path):
    # Încarcarea modelului EfficientNet-B0
    model = tf.keras.models.load_model(model_path, custom_objects={'F1_score':'F1_score'})

    # Încarcarea și pregătirea imaginii
    img = image.load_img(test_image_path, target_size=(224, 224))
    img_array = image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)
    img_array = preprocess_input(img_array)

    # Realizarea predicției
    predictions = model.predict(img_array)
    predicted_class = np.argmax(predictions, axis=1)[0]

    # Filtrarea DataFrame-ului pentru a obține rândul care corespunde clasei prezise
    filtered_data = data[data['class_id'] == predicted_class]

    # Obținerea informației din coloana 'labels' pentru acest rând
    label_info = filtered_data.iloc[0]['labels']

    # Obținerea informației din coloana 'scientific_name' pentru acest rând  
    latin_info = filtered_data.iloc[0]['scientific_name']

    # Obținerea informației din coloana 'habitat' pentru acest rând  
    habitat_info = filtered_data.iloc[0]['habitat']

    # Obținerea informației din coloana 'additional_info' pentru acest rând  
    additional_info = filtered_data.iloc[0]['additional_info']

    return predicted_class, label_info, latin_info, habitat_info, additional_info

# Funcția pentru redarea fișierului audio
def play_sound(label_info):
    global current_audio_file, is_playing
    audio_folder = '../Desktop/12/audio'
    audio_file = os.path.join(audio_folder, f"{label_info}.mp3")
    audio_file2 = os.path.join(audio_folder, f"{label_info}.ogg")
    if os.path.exists(audio_file):
        if current_audio_file != audio_file:
            current_audio_file = audio_file
            is_playing = False              # Resetarea stării dacă se încarcă alt fișier audio
        
        if is_playing:
            pygame.mixer.music.stop()
            is_playing = False
        else:
            pygame.mixer.music.load(audio_file)
            pygame.mixer.music.play()
            is_playing = True

    if os.path.exists(audio_file2):
        if current_audio_file != audio_file2:
            current_audio_file = audio_file2
            is_playing = False              # Resetarea stării dacă se încarcă alt fișier audio
        
        if is_playing:
            pygame.mixer.music.stop()
            is_playing = False
        else:
            pygame.mixer.music.load(audio_file2)
            pygame.mixer.music.play()
            is_playing = True


# Funcția pentru căutarea folderului cu imagini
def cautare_director():
    global director_selectat
    director_selectat = filedialog.askdirectory()
    if director_selectat:
        afiseaza_imagini(director_selectat)

# Funcția pentru afișarea listei de imagini
def afiseaza_imagini(director):
    extensii_imagini = ('.jpg', '.jpeg', '.png')
    imagini = []
    for root, dirs, files in os.walk(director):
        for file in files:
            if file.lower().endswith(extensii_imagini):
                imagini.append(os.path.join(root, file))
    
    lista_imagini.delete(0, tk.END)  # Șterge orice elemente din listă
    for imagine in imagini:
        lista_imagini.insert(tk.END, os.path.basename(imagine))
        lista_imagini.config(font=('Calibri', 24))

    # Activarea butonului "Afiseaza Imagine"
    frame_buton_afisare_imagine.grid()
    buton_afisare_imagine.config(state=tk.NORMAL)


# Funcția pentru afișarea imaginii selectate
def afiseaza_imagine_selectata():

    # Ștergerea tuturor elementelor din Treeview
    for item in tree.get_children():
        tree.delete(item)

    indice_selectat = lista_imagini.curselection()
    if indice_selectat:
        nume_imagine = lista_imagini.get(indice_selectat[0])
        global cale_imagine
        cale_imagine = os.path.join(director_selectat, nume_imagine)

        # Afișarea progress bar-ului
        progress_bar.grid(row=3, column=0, pady=10)
        progress_bar['value'] = 0

        # Crearea și pornirea unui thread pentru a realiza predicția
        thread_predictie = threading.Thread(target=realizeaza_predictia, args=(cale_imagine,))
        thread_predictie.start()

        # Actualizarea progress bar-ului în timp ce predicția este realizată
        actualizeaza_progress_bar(0)

# Încărcarea modelului și realizarea predicției
def realizeaza_predictia(cale_imagine):
    global predictions, label_info, latin_info, habitat_info, additional_info
    # Realizează predicția asupra imaginii
    predictions, label_info, latin_info, habitat_info, additional_info = load_and_test_model(model_path, cale_imagine)
    

    # Actualizarea UI-ului principal după afisarea rezultatelor
    root.after(0, afisare_rezultate)


# Funcția ce face tabelul și afișeaza informațille găsite în baza de date
def afisare_rezultate():
    categories = ["Specia", "Denumirea științifică", "Se găsește în", "Alte informații"]
    info = [label_info, latin_info, habitat_info, additional_info]
    # Adăugarea informațiilor în tabel
    for category,data in zip(categories, info):
        tree.insert("", "end", text=category, values=(data,))

    apply_styles(tree) # Aplicarea stilului pentru tabel
    
    # Afișarea imaginii
    incarca_si_afiseaza_imaginea(cale_imagine)

    # Oprirea și ascunderea progress bar-ului după ce imaginea este încărcată
    progress_bar.grid_forget()

    # Afișează tabel
    tree.pack(expand=True, fill=tk.X)

    play_button.pack_forget()
    audio_folder = '../Desktop/12/audio'
    audio_file = os.path.join(audio_folder, f"{label_info}.mp3")
    audio_file2 = os.path.join(audio_folder, f"{label_info}.ogg")

    if os.path.exists(audio_file):
        play_button.pack(side=tk.TOP, padx=10, pady=10)

    if os.path.exists(audio_file2):
    # Plasarea butonului "Play/Stop" sub tabel
        play_button.pack(side=tk.TOP, padx=10, pady=10)

# Funcția ce actualizează progress bar-ul
def actualizeaza_progress_bar(valoare):
    if valoare < 100:
        progress_bar['value'] = valoare
        root.after(50, actualizeaza_progress_bar, valoare + 2)

# Încărcarea și afișarea imaginii selectate
def incarca_si_afiseaza_imaginea(cale_imagine):
    if os.path.isfile(cale_imagine):
        imagine = Image.open(cale_imagine)
        imagine = ImageTk.PhotoImage(imagine)
        label_imagine.config(image=imagine)
        label_imagine.image = imagine  # Păstrează o referință pentru a evita eliberarea memoriei
    else:
        print(f"Eroare: Fișierul {cale_imagine} nu a fost găsit.") 

# Definirea stilului pentru tabel
def apply_styles(tree):
    style = ttk.Style()
    style.configure("Treeview.Heading", font=('Calibri', 28, 'bold'))  # Stil pentru antete
    style.configure("Treeview", font=('Calibri', 24))  # Stil pentru elementele din treeview
    style.configure("Treeview", rowheight=40)

# Crearea ferestrei principale
root = tk.Tk()
root.title("Aplicatie")
root.minsize(960, 640)
root.config(bg="#8d99ae")

# Configurarea fontului pentru tabel
custom_font = tkfont.Font(family="Calbri", size=24)
first_row_font = tkfont.Font(family="Calibri", size=28, weight="bold")

# Crearea frame-ului pentru butonul "Play/Stop audio"
frame_butoane = tk.Frame(root)
frame_butoane.grid(row=0, column=2, padx=10, pady=10, sticky="nsew")

# Crearea frame-ului principal pentru partea stanga
frame_stanga = tk.Frame(root)
frame_stanga.grid(row=0, column=0, padx=10, pady=10, sticky="nsew")

# Crearea butonului de căutare a directorului
buton_cautare = tk.Button(frame_stanga, text="Cauta", command=cautare_director)
buton_cautare.grid(row=0, column=0, pady=5)
buton_cautare.config(font=("Calbri", 24))

# Crearea frame-ului pentru lista de imagini si scrollbar
frame_lista_imagini = tk.Frame(frame_stanga)
frame_lista_imagini.grid(row=1, column=0, pady=10)

# Crearea listei pentru afișarea numelor fișierelor de imagine
lista_imagini = tk.Listbox(frame_lista_imagini)
lista_imagini.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)

# Crearea scrollbar-ului pentru lista de imagini
scrollbar = tk.Scrollbar(frame_lista_imagini, orient=tk.VERTICAL)
scrollbar.pack(side=tk.RIGHT, fill=tk.Y)
scrollbar.config(command=lista_imagini.yview)
lista_imagini.config(yscrollcommand=scrollbar.set)

# Crearea frame-ului pentru butonul "Afișează Imagine"
frame_buton_afisare_imagine = tk.Frame(frame_stanga)
frame_buton_afisare_imagine.grid(row=2, column=0, pady=5)
frame_buton_afisare_imagine.grid_remove()  # Ascunde inițial frame-ul

# Crearea butonului pentru afișarea imaginii selectate
buton_afisare_imagine = tk.Button(frame_buton_afisare_imagine, text="Afiseaza Imagine", command=afiseaza_imagine_selectata, state=tk.DISABLED)
buton_afisare_imagine.pack(anchor=tk.CENTER)
buton_afisare_imagine.config(font=("Calibri", 24))

# Crearea progress bar-ului
progress_bar = ttk.Progressbar(frame_stanga, orient=tk.HORIZONTAL, mode='determinate', maximum=100)

# Crearea frame-ului pentru afișarea imaginii în partea dreaptă
frame_dreapta = tk.Frame(root)
frame_dreapta.grid(row=0, column=1, pady=10, sticky="nsew")

# Setări tabel
tree = ttk.Treeview(frame_dreapta)
style = ttk.Style()
tree["columns"] = ("1")
tree.column("#0", width=400, minwidth=400, stretch=tk.NO)
tree.column("1", width=400, minwidth=400, stretch=tk.YES)  # Permit extinderea coloanelor
tree.heading("#0", text="Categorie")
tree.heading("1", text="Informatie")
tree.pack_forget()


# Extinderea frame-ului din dreapta să ocupe spațiul rămas
root.grid_columnconfigure(1, weight=1)
root.grid_rowconfigure(0, weight=1)

# Eticheta pentru afișarea imaginii selectate
label_imagine = tk.Label(frame_dreapta)
label_imagine.pack(pady=(50, 10))  # Adaugăm un padding suplimentar între imagine și marginea de sus a ferestrei

# Crearea butonului "Play/Stop"
play_button = tk.Button(frame_dreapta, text="Play/Stop audio", command=lambda: play_sound(label_info))
play_button.config(font=("Calbri", 24))

# Directorul selectat
director_selectat = ""
cale_imagine = ""  # Variabilă globală pentru calea imaginii selectate

# Rularea buclei principale a aplicației
root.mainloop()