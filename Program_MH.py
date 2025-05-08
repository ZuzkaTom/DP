import numpy as np  
import matplotlib.pyplot as plt  
import librosa  
from matplotlib.widgets import Button  
from tqdm import tqdm  
from scipy.signal import filtfilt 

# Nacitanie suboru
test_signal, sr = librosa.load("trubka.wav", sr=20000)
#test_signal, sr = librosa.load("bowls.wav", sr=20000)

# Orezanie signalu na 1 sekundu
test_signal = test_signal[:sr]

# Parametre signalu
sr = 20000  # vzorkovacia frekvencia (20 kHz)
duration = 1.0  # trvanie signalu v sekundach
vzorky = 500 # pocet vzoriek pre vykreslovanie
dt = 1 / sr  # Δt: vzorkovaci krok v sekundach
t = np.arange(0, duration, dt)

# Generovanie umeleho signalu    
freq = 200
#test_signal = np.sin(2 * np.pi * freq * t) + np.sin(2 * np.pi * 2 * freq * t) + np.sin(2 * np.pi * 3 * freq * t)
#test_signal = np.sin(2 * np.pi * 600 * t) + np.sin(2 * np.pi * 1500 * t)+np.sin(2 * np.pi * 2400 * t)


# Digitalny pasmovy filter s koeficientmi
def bandpass_filter(x):
    b = [0.8878, 0, -0.8878]
    a = [1, 0.2243, -0.7757]
    return filtfilt(b, a, x)  

# Aplikujeme digitalny pasmovy filter na testovaci signal 
filtered_signal = bandpass_filter(test_signal)
print("Efekty vonkajsieho a stredneho ucha : bandpass filter")

# Funkcia na vypocet ERB
def erb_scale(frequency):
    return 24.7 * (4.37e-3 * frequency + 1)

# Funkcia na prevod frekvencie (v Hz) do ERB-rate skaly
def hz_to_erb_rate(frequency):
    return 21.4 * np.log10(4.37e-3 * frequency + 1)

# Inverzna funkcia: prevod z ERB-rate spat na frekvenciu v Hz
def erb_rate_to_hz(erb_rate):
    return (10**(erb_rate / 21.4) - 1) / 4.37e-3

# Vypocet centralnych frekvencii pre filterbank
def erb_center_frequencies(start_freq=80, end_freq=8000, erb_step=0.25):
    erb_start = hz_to_erb_rate(start_freq)  
    erb_end = hz_to_erb_rate(end_freq)      
    erb_points = np.arange(erb_start, erb_end + erb_step, erb_step)  
    return erb_rate_to_hz(erb_points)       

# Implementacia gammatonoveho filterbanku pomocou IIR filtracie
def gammatone_filterbank_iir(signal, center_freqs, order=4):
    
    filtered_signals = []  

    for f in tqdm(center_freqs, desc="Filtracia bazilarnou membranou : gammatone filterbank"):
        z = np.zeros_like(signal, dtype=np.complex128)

        for k in range(len(signal)):
            z[k] = signal[k] * np.exp(-1j * 2 * np.pi * f * t[k])

        alpha = 1 - np.exp(-2 * np.pi * erb_scale(f) * dt)
        w = np.zeros_like(z, dtype=np.complex128)

        for k in range(1, len(z)):
            w[k] = w[k-1] + alpha * (z[k-1] - w[k-1])

        for _ in range(order - 1):
            w2 = np.zeros_like(w, dtype=np.complex128)

            for k in range(1, len(w)):
                w2[k] = w2[k-1] + alpha * (w[k-1] - w2[k-1])
            w = w2

        y = np.real(w * np.exp(1j * 2 * np.pi * f * t))

        filtered_signals.append(y)

    return np.array(filtered_signals)

# Vypocet centralnych frekvencii filtrov 
center_freqs = erb_center_frequencies(start_freq=80, end_freq=8000, erb_step=0.25)
print(f"Pocet filtrov: {len(center_freqs)}")

# Aplikacia IIR gammatonoveho filterbanku – kazdy filter simuluje jeden kanal bazilarnej membrany
gammatone_signals = gammatone_filterbank_iir(filtered_signal, center_freqs)

# Mechanicko-nervova transdukcia
# Parametre modelu
M = 1.0  # Maximalne mnozstvo volneho neurotransmitera
A = 5  # Konstanta pre permeabilitu
B = 300  # Konstanta pre permeabilitu
g = 2000  # Konstanta pre permeabilitu
y = 5.05  # Rychlost doplnania neurotransmitera
L = 2500  # Rychlost strat zo synaptickej strbiny
r = 6580  # Rychlost navratu neurotransmitera do recyklacneho zasobneho priestoru
x = 66.31  # Rychlost uvolnenia neurotransmitera z recyklacneho zasobneho priestoru
h = 50000  # Konstanta pre pravdepodobnost spiku

# Parametre pre vypocet amplitudy stimulu
dB = 80  # Decibelova uroven signalu
amp = 10 ** (dB / 20 - 1.35)  # Amplituda signalu
endsilence = 0.01  # Trvanie pociatocneho ticha v sekundach

#Vahova funkcia pre refraktalne efekty
def W_function(t):
    return np.where((t > 0) & (t < 0.001), 1.0, 0.0)

# Vypocet pociatocnych podmienok pre q, c, w
kt_spontaneous = (g * A) / (A + B)  # Permeabilita pri spontannej aktivite
c_spontaneous = (M * y * kt_spontaneous) / (L * kt_spontaneous + y * (L + r))  # Mnozstvo neurotransmitera v synaptickej strbine
q_spontaneous = (c_spontaneous * (L + r)) / kt_spontaneous  # Mnozstvo volneho neurotransmitera
w_spontaneous = (c_spontaneous * r) / x  # Mnozstvo neurotransmitera v recyklacnom zasobnom priestore
 
# Pole na okladanie vysledkov
kt_values = []
q_values = []
cleft_values = []
p_spike_values = []

for sig_idx, sig in enumerate(tqdm(gammatone_signals, desc="Mechanicko-nervova transdukcia a refrakterne efekty: simulacia Meddis-Hewitt")):
    # Resetovanie hodnot pre kazdy kanal
    spike_history = np.zeros_like(t)
    q = q_spontaneous  # Pociatocne mnozstvo volneho neurotransmitera
    c = c_spontaneous  # Pociatocne mnozstvo neurotransmitera v synaptickej strbine
    w = w_spontaneous  # Pociatocne mnozstvo neurotransmitera v recyklacnom zasobnom priestore
    
    # Pole na ukladanie vysledkov pre kazdy kanal
    kt_values_channel = []
    q_values_channel = []
    cleft_values_channel = []
    p_spike_values_channel = []
    
    # Pre kazdy casovy krok simulacie
    for i, t_sim in enumerate(t):
        # Vyberieme jednu vzorku zo signalu pre cas t_sim
        stimulus = sig[int(t_sim * sr)] 

        # Vypocet amplitudy stimulu
        if t_sim > endsilence:
            stimulus = amp * stimulus 
        else:
            stimulus = 0

        # Vypocet permeability membrany
        if stimulus + A > 0:
            kt = (g * dt * (stimulus + A)) / (stimulus + A + B)
        else:
            kt = 0

        # Vypocet zmien v mnozstvach neurotransmitera
        if M >= q:
            replenish = y * dt * (M - q)  # Doplnanie volneho neurotransmitera
        else: 
            replenish = 0
        eject = kt * q  # Uvolnenie neurotransmitera do synaptickej strbiny
        loss = L * dt * c  # Strata neurotransmitera zo synaptickej strbiny
        reuptake = r * dt * c  # Navrat neurotransmitera do recyklacneho priestoru
        reprocess = x * dt * w  # Presun neurotransmitera z recyklacneho priestoru do zasobneho priestoru

        # Aktualizacia 
        q = q + replenish - eject + reprocess
        c = c + eject - loss - reuptake
        w = w + reuptake - reprocess

        # Vypocet pravdepodobnosti spiku 
        p_spike = h * c * dt  
        
        # Výpočet vplyvu predchádzajúcich spikov pomocou váhovej funkcie
        ref_effect = np.sum(spike_history[:i] * W_function(t_sim - t[:i]))
        ref_effect = np.clip(ref_effect, 0, 1)

        # Úprava pravdepodobnosti spiku na základe refraktérneho efektu
        p_spike_adjusted = p_spike * (1 - ref_effect)

        # Rozhodnutie o spiku na základe upravenej pravdepodobnosti
        if np.random.random() < p_spike_adjusted:
            spike_history[i] = 1

        # Ulozenie vysledkov do zoznamov pre aktualny kanal
        kt_values_channel.append(kt)  # Ukladame hodnotu kt 
        q_values_channel.append(q)  # Ukladame hodnotu q 
        cleft_values_channel.append(c)  # Ukladame koncentraciu neurotransmitera v synaptickej strbine
        p_spike_values_channel.append(p_spike_adjusted)  # Ulozena upravena pravdepodobnost spiku

    # Ulozenie vysledkov pre kazdy kanal (pridanie do globalnych zoznamov)
    kt_values.append(kt_values_channel)
    q_values.append(q_values_channel)
    cleft_values.append(cleft_values_channel)
    p_spike_values.append(p_spike_values_channel)

# Vypocet autokorelacnych histogramov
# Casova konstanta pouzita vo vahovacej funkcii e^(-T/τ)
time_constant = 0.0025  

# Dlzka, pocas ktorej sa beru do uvahy predchadzajuce spiky
acf_window = 0.0075     

# Vytvorenie vsetkych posunov δt, pre ktore budeme pocitat autokorelaciu
delta_t_values = np.arange(0.00005, 0.01667, dt)  

# Pocet δt hodnot = velkost histogramu pre kazdy kanal
num_bins = len(delta_t_values)  

# Zoznam, do ktoreho budeme ukladat histogram pre kazdy kanal
histograms = []  

# Hlavny cyklus cez kanaly
for k, p_t in enumerate(tqdm(p_spike_values, desc="Distribucia casovych intervalov : vypocet histogramov")):  
    histogram = np.zeros(num_bins)      
    p_t = np.array(p_t)                 

    # Cyklus cez vsetky oneskorenia δt
    for i, delta in enumerate(tqdm(delta_t_values, desc=f"δt pre kanal {k+1}", leave=False)):  
        # Vyjadrime δt ako pocet vzoriek
        delay_samples = int(delta / dt)     
        # Urcime, kolko vzoriek T mame v intervale 0–7.5 ms    
        max_T = int(acf_window / dt)            

        # Cyklus cez vsetky casy T
        for T in range(1, max_T): 
            # Berieme len tie t, pri ktorych vieme bezpecne siahnut do minulosti az o T+δt, teda:
            t_idx = np.arange(delay_samples + T, len(p_t))

            # Vypocitame indexy pre p(t−T)
            t_minus_T = t_idx - T

            # Vypocitame indexy pre p(t−T−δt)
            t_minus_T_minus_delta = t_idx - T - delay_samples

            # Zabezpecime, ze indexy neprecitaju data mimo pola
            valid = (t_minus_T >= 0) & (t_minus_T_minus_delta >= 0)

            if np.any(valid):  # Pokracujeme len vtedy, ked mame aspon jeden platny index
                # Extrahujeme hodnoty pravdepodobnosti spiku v case t−T a t−T−δt
                pit_T = p_t[t_minus_T[valid]]              # p(t−T)
                pit_T_dt = p_t[t_minus_T_minus_delta[valid]]  # p(t−T−δt)

                # Vypocitame vahu podla vzdialenosti T – Lickliderov exponencialny utlm
                weights = np.exp(-T * dt / time_constant)  # e^(−T/τ)

                # Vypocitame prispevok do histogramu pre dany δt:
                histogram[i] += np.sum(pit_T * pit_T_dt * weights)

    # Po dokonceni cyklu cez δt ulozime histogram pre aktualny kanal
    histograms.append(histogram)

# Vypocet SACF ako priemer histogramov
time_intervals = delta_t_values
sacf = np.mean(histograms, axis=0)
print("Scitanie napriec kanalmi : SACF")

# casove hranice pre vypocet fundamentalnej frekvencie
min_lag = 0.001      # 1 ms = 1000 Hz
max_lag = 0.0125     # 12.5 ms = 80 Hz

# Vyfiltrujeme hodnoty SACF v danom intervale oneskoreni
mask = (delta_t_values >= min_lag) & (delta_t_values <= max_lag)
sacf_in_range = sacf[mask]
delta_in_range = delta_t_values[mask]

# Najdeme index najvyssieho vrcholu v tomto intervale
peak_idx = np.argmax(sacf_in_range)
dominant_delta = delta_in_range[peak_idx]
f0_estimated = 1 / dominant_delta

print(f"Extrakcia vysky tonu : odhadnuta fundamentalna frekvencia (f₀): {f0_estimated:.2f} Hz")

# Vykreslovanie 

# Zoznam grafov, ktore chceme prepinat
graphs = [
    (test_signal, "Vstupny signal"),
    (filtered_signal, "Filtrovany signal"),
    (gammatone_signals, "Vystup gamatonovej banky filtrov"),
    (cleft_values, "Mnozstvo neurotransmitera c(t) v synaptickej strbine"),
    (q_values, "Mnozstvo neurotransmitera q(t) v zasobniku volneho neurotransmitera"),
    (p_spike_values, "Pravdepodobnost spiku p_spike(t)"),
    (histograms, "Autokorelacne histogramy"),
    (sacf, "Suhrnny autokorelacny histogram (SACF)")
]

# Index aktualneho kanala pre histogramy
current_histogram_channel = 0 
# Index aktualne zobrazovaneho grafu
current_graph = 0  

# Vytvorenie hlavneho grafu
fig, ax = plt.subplots(figsize=(10, 4))  
# Posunutie plochy grafu vyssie, aby bolo miesto pre tlacidla
plt.subplots_adjust(bottom=0.2)  

# Funkcia na aktualizaciu grafu
def update_graph(idx):
    # Budeme pracovat s globalnou premennou current_graph, a nie vytvarat novu lokalnu premennu
    global current_graph 
    # Zabezpeci cyklicke prepinanie grafov (% modulo)
    current_graph = idx % len(graphs)  
    ax.clear()  
    ax.set_title(graphs[current_graph][1])  
    ax.set_xlabel("cas (s)")  
    ax.set_ylabel("Amplituda")  

    # Ak zobrazujeme vystup gammatone bankz filtrov
    if current_graph == 2:  
        # Vykreslujeme len kazdy druhy kanal kvoli prehladnosti
        selected_indices = np.arange(0, len(center_freqs), 2)  
        # Vertikalny rozostup
        spacing = 0.1  
        # Amplitudove skalovanie
        scale_factor = 2.0  # 5 pre trubku, 20 pre bowls
        # Rovnomerne vizualne rozostupy
        offsets = np.arange(len(selected_indices)) * spacing
        # Vyplname po okraj
        min_fill = offsets[0] - spacing  

        for idx, i in enumerate(selected_indices):
            sig = graphs[current_graph][0][i]
            scaled_sig = sig * scale_factor
            y_vals = scaled_sig[-vzorky:] + offsets[idx]
            ax.fill_between(t[-vzorky:], y_vals, min_fill, linewidth=0.8, edgecolor='black', zorder=-i)

        # Nastavenie osi
        ax.set_yticks(offsets)
        yticks_labels = [f"{center_freqs[i]:.0f} Hz" if idx % 3 == 0 else "" for idx, i in enumerate(selected_indices)]
        ax.set_yticklabels(yticks_labels, fontsize=14)
        ax.set_ylabel("Centralne frekvencie (Hz)")
        ax.set_xlabel("cas (s)")
        ax.set_title("Vystupy suboru gamatonovych filtrov")

    # c(t) - mnozstvo neurotransmitera v synaptickej strbine
    elif current_graph in [3]:  
        data = graphs[current_graph][0]
        title = graphs[current_graph][1]

        # Najde extremov napriec kanalmi
        cleft_min = min(np.min(cleft[-vzorky:]) for cleft in cleft_values) 
        cleft_max = max(np.max(cleft[-vzorky:]) for cleft in cleft_values)

        values = data[current_histogram_channel][-vzorky:]  
        ax.plot(t[-vzorky:], values)

        ax.set_title(f"{title} – kanal {current_histogram_channel + 1} ({center_freqs[current_histogram_channel]:.1f} Hz)")
        ax.set_xlabel("cas (s)")
        ax.set_ylabel("c(t)")  
        # Nastavime mierku pre kazdy kanal
        ax.set_ylim(cleft_min-0.001, cleft_max+0.005) 
        ax.grid(True)

    # q(t) - mnozstvo neurotransmitera v zasobniku volneho neurotransmitera
    elif current_graph in [4]:  
        data = graphs[current_graph][0]
        title = graphs[current_graph][1]

        # Najde extremov napriec kanalmi
        q_min = min(np.min(q[-vzorky:]) for q in q_values)
        q_max = max(np.max(q[-vzorky:]) for q in q_values)

        values = data[current_histogram_channel][-vzorky:]  
        ax.plot(t[-vzorky:], values)

        ax.set_title(f"{title} – kanal {current_histogram_channel + 1} ({center_freqs[current_histogram_channel]:.1f} Hz)")
        ax.set_xlabel("cas (s)")
        ax.set_ylabel("q(t)") 
        # Nastavime mierku pre kazdy kanal
        ax.set_ylim(q_min-0.001, q_max+0.005) 
        ax.grid(True)

    # p_spike(t)
    elif current_graph in [5]:  
        data = graphs[current_graph][0]
        title = graphs[current_graph][1]

        # Najde extremov napriec kanalmi
        p_min = min(np.min(p[-vzorky:]) for p in p_spike_values)
        p_max = max(np.max(p[-vzorky:]) for p in p_spike_values)

        values = data[current_histogram_channel][-vzorky:]  
        ax.plot(t[-vzorky:], values)

        ax.set_title(f"{title} – kanal {current_histogram_channel + 1} ({center_freqs[current_histogram_channel]:.1f} Hz)")
        ax.set_xlabel("cas (s)")
        ax.set_ylabel("p_spike(t)") 
        # Fixneme mierku pre kazdy kanal
        ax.set_ylim(p_min-0.001, p_max+0.005) 
        ax.grid(True)

    # Autokorelacny histogram
    elif current_graph == 6:  
        # Vypocita najvacsiu hodnotu zo vsetkych histogramov
        hist_max = max(np.max(h) for h in histograms) 

        ax.bar(delta_t_values, histograms[current_histogram_channel], width=dt, color='teal', alpha=0.7)
        ax.set_title(f"Autokorelacny histogram – kanal {current_histogram_channel + 1} ({center_freqs[current_histogram_channel]:.1f} Hz)")
        ax.set_xlabel("casovy posun (s)")
        ax.set_ylabel("Vazeny sucet korelacnych hodnot")
        ax.set_ylim(0, hist_max * 1.05)  # +5 % rezerva
        ax.grid(True)

    # Ak zobrazujeme suhrnny autokorelacny histogram (SACF)
    elif current_graph == 7:    
        ax.bar(delta_t_values, graphs[current_graph][0], width=dt, color='purple', label="SACF (histogram)")
        ax.axvline(x=dominant_delta, color='orange', linestyle='--', label=f"Odhad f₀ = {f0_estimated:.1f} Hz")
        ax.set_xlabel("casovy posun (s)")
        ax.set_ylabel("Vazeny sucet korelacnych hodnot")
        ax.set_title("Suhrnny autokorelacny histogram (SACF) s oznacenymi vrcholmi")
        ax.legend()
        ax.grid(True)

    # Pre povodny a filtrovany signal
    else:  
        ax.plot(t[-vzorky:], graphs[current_graph][0][-vzorky:])  

    # Obnovenie zobrazenia grafu
    fig.canvas.draw()  

# Funkcia, ktora meni index aktualne zobrazovaneho kanala histogramu
def change_hist_channel(step): 
    global current_histogram_channel
    current_histogram_channel = (current_histogram_channel + step) % len(histograms)
    # Znovu vykresli graf s novym kanalom histogramu 
    update_graph(current_graph) 

# Prve zobrazenie grafu
update_graph(current_graph)

# Tlacidlo "dalsi"
ax_next = plt.axes([0.8, 0.05, 0.1, 0.075])  
btn_next = Button(ax_next, 'Dalsi')  
# Funkcia na prepnutie na dalsi graf
btn_next.on_clicked(lambda event: update_graph(current_graph + 1))  

# Tlacidlo "Predchadzajuci"
ax_prev = plt.axes([0.65, 0.05, 0.1, 0.075])  
btn_prev = Button(ax_prev, 'Predch.')  
# Funkcia na prepnutie na predchadzajuci graf
btn_prev.on_clicked(lambda event: update_graph(current_graph - 1))  

# Tlacidlo "dalsi kanal pre histogram"
ax_next_hist = plt.axes([0.45, 0.05, 0.1, 0.075])
btn_next_hist = Button(ax_next_hist, 'Kanal +')
btn_next_hist.on_clicked(lambda event: change_hist_channel(1))

# Tlacidlo "Predchadzajuci kanal pre histogram"
ax_prev_hist = plt.axes([0.3, 0.05, 0.1, 0.075])
btn_prev_hist = Button(ax_prev_hist, 'Kanal -')
btn_prev_hist.on_clicked(lambda event: change_hist_channel(-1))

# Zobrazenie grafov
plt.show()


