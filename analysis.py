import pandas as pd
import numpy as np
import torch
from transformers import BertTokenizer, BertModel
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
import seaborn as sns
import warnings

# Αγνοούμε τις προειδοποιήσεις για πιο καθαρό output
warnings.filterwarnings("ignore")

def get_bert_embedding(text: str, model: BertModel, tokenizer: BertTokenizer) -> np.ndarray:
    """
    Υπολογίζει το embedding ενός κειμένου χρησιμοποιώντας το BERT.
    Παίρνουμε τον μέσο όρο των last hidden states όλων των tokens.
    """
    if not isinstance(text, str) or pd.isna(text) or not text.strip():
        return np.zeros(model.config.hidden_size)
        
    inputs = tokenizer(text, return_tensors='pt', truncation=True, max_length=512, padding=True)
    with torch.no_grad():
        outputs = model(**inputs)
    
    embedding = outputs.last_hidden_state.mean(dim=1).squeeze().numpy()
    return embedding

def calculate_cosine_similarities(df: pd.DataFrame, model: BertModel, tokenizer: BertTokenizer) -> pd.DataFrame:
    """
    Υπολογίζει την ομοιότητα συνημιτόνου μεταξύ του Original και των Reconstructed κειμένων.
    Λειτουργεί στο αρχικό 'wide' DataFrame.
    """
    print("Υπολογισμός ομοιότητας συνημιτόνου...")
    
    embedding_cache = {}
    
    text_columns_to_process = [
        'Original_Text', 
        'Reconstructed_LangTool', 
        'Reconstructed_T5',
        'Reconstructed_Spacy',
        'Reconstructed_Custom'
    ]

    # Υπολογίζουμε τα embeddings για όλες τις σχετικές στήλες και τα αποθηκεύουμε
    for col in text_columns_to_process:
        if col in df.columns:
            embedding_cache[f'Embedding_{col}'] = df[col].apply(lambda x: get_bert_embedding(x, model, tokenizer))
            
    # Υπολογισμός similarities χρησιμοποιώντας τα cached embeddings
    original_embeddings = embedding_cache.get('Embedding_Original_Text')
    
    if original_embeddings is None:
        print("Προσοχή: Δεν βρέθηκε η στήλη 'Embedding_Original_Text' για τον υπολογισμό similarities.")
        return df

    reconstructed_cols = [col for col in text_columns_to_process if 'Reconstructed' in col]

    for recon_text_col in reconstructed_cols:
        recon_embedding_col_name = f'Embedding_{recon_text_col}'
        if recon_embedding_col_name in embedding_cache:
            sim_col_name = f"CosineSim_{recon_text_col}"
            
            reconstructed_embeddings = embedding_cache[recon_embedding_col_name]
            
            similarities = [
                cosine_similarity(orig.reshape(1, -1), recon.reshape(1, -1))[0][0]
                if isinstance(orig, np.ndarray) and isinstance(recon, np.ndarray) and not np.all(recon == 0)
                else np.nan
                for orig, recon in zip(original_embeddings, reconstructed_embeddings)
            ]
            df[sim_col_name] = similarities
            
    return df


if __name__ == "__main__":
    print("Ξεκινά η διαδικασία υπολογιστικής ανάλυσης...")

    # --- Βήμα 1: Φόρτωση Δεδομένων & Μοντέλου ---
    try:
        df_wide = pd.read_csv("reconstructions.csv")
    except FileNotFoundError:
        print("Σφάλμα: Το αρχείο 'reconstructions.csv' δεν βρέθηκε. Βεβαιωθείτε ότι έχετε εκτελέσει πρώτα το reconstruction.py")
        exit()

    print("Φόρτωση μοντέλου BERT (bert-base-uncased)...")
    try:
        tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
        model = BertModel.from_pretrained('bert-base-uncased')
    except Exception as e:
        print(f"Σφάλμα κατά τη φόρτωση του μοντέλου BERT: {e}")
        exit()

    # --- Βήμα 1.5: Υπολογισμός Cosine Similarity (στο wide format) ---
    analysis_results_df = calculate_cosine_similarities(df_wide.copy(), model, tokenizer)


    # --- Βήμα 2: Ριζική Αναδιάρθρωση Δεδομένων (Wide to Long) ---
    print("Αναδιάρθρωση δεδομένων από 'wide' σε 'long' format (melt)...")
    
    id_vars = ['Original_Text_Name']
    
    value_vars = [
        'Original_Text',
        'Reconstructed_LangTool',
        'Reconstructed_T5',
        'Reconstructed_Spacy',
        'Reconstructed_Custom'
    ]
    
    df_long = df_wide.melt(
        id_vars=id_vars, 
        value_vars=value_vars, 
        var_name='method_raw',
        value_name='text'
    )
    
    df_long.dropna(subset=['text'], inplace=True)
    df_long = df_long[df_long['text'].str.strip() != '']

    method_map = {
        'Original_Text': 'Original',
        'Reconstructed_LangTool': 'LangTool',
        'Reconstructed_T5': 'T5',
        'Reconstructed_Spacy': 'Spacy',
        'Reconstructed_Custom': 'Custom'
    }
    df_long['method'] = df_long['method_raw'].replace(method_map)
    
    print("Η αναδιάρθρωση ολοκληρώθηκε. Δείγμα των νέων δεδομένων:")
    print(df_long[['Original_Text_Name', 'method']].head())
    print(f"\nΣυνολικά σημεία για ανάλυση: {len(df_long)}")

    # --- Βήμα 3: Υπολογισμός Embeddings (στο long format) ---
    print("Υπολογισμός embeddings για το αναδιαρθρωμένο DataFrame...")
    df_long['embedding'] = df_long['text'].apply(lambda x: get_bert_embedding(x, model, tokenizer))

    # --- Βήμα 4: Μείωση Διαστατικότητας (PCA & t-SNE) ---
    X = np.vstack(df_long['embedding'].values)

    print("Εκτέλεση PCA...")
    pca = PCA(n_components=2, random_state=42)
    X_pca = pca.fit_transform(X)
    df_long['pca1'] = X_pca[:, 0]
    df_long['pca2'] = X_pca[:, 1]

    print("Εκτέλεση t-SNE (μπορεί να αργήσει)...")
    perplexity = min(5, len(X) - 1)
    
    tsne = TSNE(n_components=2, random_state=42, perplexity=perplexity, max_iter=1000)
    X_tsne = tsne.fit_transform(X)
    df_long['tsne1'] = X_tsne[:, 0]
    df_long['tsne2'] = X_tsne[:, 1]
    
    # --- Βήμα 5: Οπτικοποίηση (Με Απόλυτο Έλεγχο) ---
    print("Δημιουργία γραφημάτων...")
    
    method_order = ['Original', 'LangTool', 'T5', 'Spacy', 'Custom']
    color_palette = {
        'Original': 'tab:blue',
        'LangTool': 'tab:orange',
        'T5': 'tab:green',
        'Spacy': 'tab:red',
        'Custom': 'tab:purple'
    }
    
    # ---- Γράφημα PCA ----
    plt.figure(figsize=(16, 9))
    sns.scatterplot(
        data=df_long, x='pca1', y='pca2', hue='method', style='Original_Text_Name',
        hue_order=method_order, palette=color_palette, s=150, alpha=0.8
    )
    plt.title('PCA of Text Embeddings - Χρώματα: Μέθοδοι, Σχήματα: Κείμενο 1 ή Κείμενο 2', fontsize=14)
    plt.xlabel('Principal Component 1', fontsize=12)
    plt.ylabel('Principal Component 2', fontsize=12)
    plt.grid(True)
    plt.legend(title='Μέθοδος | Κείμενο', bbox_to_anchor=(1.02, 1), loc='upper left', borderaxespad=0.)
    plt.tight_layout(rect=[0, 0, 0.85, 1])
    plt.savefig("pca_visualization.png", dpi=300)
    print("Το γράφημα PCA αποθηκεύτηκε στο 'pca_visualization.png'")
    plt.close()

    # ---- Γράφημα t-SNE ----
    plt.figure(figsize=(16, 9))
    sns.scatterplot(
        data=df_long, x='tsne1', y='tsne2', hue='method', style='Original_Text_Name',
        hue_order=method_order, palette=color_palette, s=150, alpha=0.8
    )
    plt.title('t-SNE of Text Embeddings - Χρώματα: Μέθοδοι, Σχήματα: Κείμενο 1 ή Κείμενο 2', fontsize=14)
    plt.xlabel('t-SNE feature 1', fontsize=12)
    plt.ylabel('t-SNE feature 2', fontsize=12)
    plt.grid(True)
    plt.legend(title='Μέθοδος | Κείμενο', bbox_to_anchor=(1.02, 1), loc='upper left', borderaxespad=0.)
    plt.tight_layout(rect=[0, 0, 0.85, 1])
    plt.savefig("tsne_visualization.png", dpi=300)
    print("Το γράφημα t-SNE αποθηκεύτηκε στο 'tsne_visualization.png'")
    plt.close()

    # --- Βήμα 6: Αποθήκευση Τελικών Αποτελεσμάτων ---
    final_df_cols = [col for col in analysis_results_df.columns if 'Embedding_' not in col]
    final_df = analysis_results_df[final_df_cols]
    final_df.to_csv("analysis_results.csv", index=False, encoding='utf-8-sig')
    
    print("\nΗ διαδικασία ανάλυσης ολοκληρώθηκε!")
    print("Τα αποτελέσματα αποθηκεύτηκαν στο 'analysis_results.csv'.")
    print("Οι οπτικοποιήσεις αποθηκεύτηκαν στα αρχεία 'pca_visualization.png' και 'tsne_visualization.png'.")