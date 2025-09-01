# NLP Assignment - Text Reconstruction

Αυτό το αποθετήριο περιέχει τον κώδικα και τα αποτελέσματα για την εργασία στο μάθημα "Επεξεργασία Φυσικής Γλώσσας". Ο στόχος του project είναι η ανακατασκευή και η υπολογιστική ανάλυση δύο αγγλικών κειμένων με γραμματικά και συντακτικά λάθη, χρησιμοποιώντας διάφορες τεχνικές NLP.

## Τι Κάνει το Project

1.  **Ανακατασκευή (Reconstruction):** Το script `reconstruction.py` παίρνει τα αρχικά κείμενα και τα ξαναγράφει με 4 διαφορετικούς τρόπους (με απλό καθαρισμό, με διόρθωση γραμματικής, με παράφραση από Τ5, και με έναν δικό μου κανόνα). Όλα τα αποτελέσματα αποθηκεύονται στο `reconstructions.csv`.
2.  **Ανάλυση (Analysis):** Το script `analysis.py` διαβάζει το CSV, μετατρέπει όλα τα κείμενα (αρχικά και διορθωμένα) σε διανύσματα (embeddings) με το BERT, και υπολογίζει πόσο μοιάζουν μεταξύ τους (cosine similarity). Στο τέλος, φτιάχνει και δύο διαγράμματα (PCA & t-SNE) για να δούμε οπτικά τις διαφορές και αποθηκεύει τα σκορ ομοιότητας στο `analysis_results.csv`.

## Setup & Εγκατάσταση

### Προαπαιτούμενα

- **Python:** Το project έχει αναπτυχθεί και δοκιμαστεί με **Python 3.12**. Βεβαιωθείτε ότι έχετε εγκατεστημένη μια συμβατή έκδοση (>=3.12, <3.13).
- **Poetry:** Το project χρησιμοποιεί **Poetry** για τη διαχείριση των πακέτων. Αν δεν το έχετε, μπορείτε να το εγκαταστήσετε ακολουθώντας τις οδηγίες [εδώ](https://python-poetry.org/docs/).

### Βήματα Εγκατάστασης

Για να το τρέξετε:

1.  **Κάντε clone το repo:**
    ```bash
    git clone https://github.com/jimsym14/nlp-assignment-2025
    cd nlp-assignment-2025
    ```
2.  **Εγκαταστήστε το Poetry:**
    ```bash
    pip3 install poetry
    ```
3.  **Εγκαταστήστε τις εξαρτήσεις με το Poetry:**
    ```bash
    poetry install
    ```
    Αυτή η εντολή θα διαβάσει το αρχείο `pyproject.toml`, θα δημιουργήσει ένα virtual environment και θα κατεβάσει αυτόματα όλες τις βιβλιοθήκες που χρειάζονται (pytorch, transformers, spacy κ.λπ.) στις σωστές εκδόσεις.

## Πώς να το Τρέξετε

Τρέχετε τα scripts μέσα από το περιβάλλον του Poetry για να είστε σίγουροι ότι χρησιμοποιούνται οι σωστές βιβλιοθήκες.

1.  **Βήμα 1: Λήψη Γλωσσικού Μοντέλου spaCy:**

    ```bash
    poetry run python -m spacy download en_core_web_sm
    ```

2.  **Βήμα 2: Τρέξτε την ανακατασκευή:**

    ```bash
    poetry run python reconstruction.py
    ```

    Αυτό μπορεί να πάρει λίγη ώρα, ειδικά την πρώτη φορά που θα κατεβάσει τα μοντέλα (T5, spacy). Όταν τελειώσει, θα έχει δημιουργηθεί το αρχείο `reconstructions.csv`.

3.  **Βήμα 3: Τρέξτε την ανάλυση:**
    ```bash
    poetry run python analysis.py
    ```
    Αυτό θα χρησιμοποιήσει το αρχείο από το προηγούμενο βήμα και θα παράξει τα αρχεία `analysis_results.csv`, `pca_visualization.png`, και `tsne_visualization.png`.

## Δομή του Project

- `reconstruction.py`: Το script για την ανακατασκευή των κειμένων.
- `analysis.py`: Το script για την ανάλυση και τη σύγκριση.
- `pyproject.toml` και `poetry.lock`: Τα αρχεία του Poetry με τις εξαρτήσεις του project.
- `README.md`: Αυτό το αρχείο.
- `reconstructions.csv`: Τα δεδομένα που παράγονται από το `reconstruction.py`.
- `analysis_results.csv`: Τα τελικά αποτελέσματα ομοιότητας από το `analysis.py`.
- `pca_visualization.png` & `tsne_visualization.png`: Οι οπτικοποιήσεις που παράγονται.
