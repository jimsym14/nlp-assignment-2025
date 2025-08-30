import pandas as pd
import spacy
import language_tool_python
from transformers import T5ForConditionalGeneration, T5Tokenizer
import torch
from tqdm import tqdm
import warnings

warnings.filterwarnings("ignore")

# --- ΑΡΧΙΚΑ ΚΕΙΜΕΝΑ ---

TEXT_1 = """Today is our dragon boat festival, in our Chinese culture, to celebrate it with all safe and great in
our lives. Hope you too, to enjoy it as my deepest wishes.

Thank your message to show our words to the doctor, as his next contract checking, to all of us.
I got this message to see the approved message. In fact, I have received the message from the
professor, to show me, this, a couple of days ago. I am very appreciated the full support of the
professor, for our Springer proceedings publication"""

TEXT_2 = """During our final discuss, I told him about the new submission the one 
we were waiting since last autumn, but the updates was confusing as it not included the full feedback from reviewer or
maybe editor?

Anyway, I believe the team, although bit delay and less communication at recent days, they really
tried best for paper and cooperation. We should be grateful, I mean all of us, for the acceptance
and efforts until the Springer link came finally last week, I think.
Also, kindly remind me please, if the doctor still plan for the acknowledgments section edit before
he sending again.
Because I didn't see that part final yet, or maybe I missed, I apologize if so.
Overall, let us make sure all are safe and celebrate the outcome with strong coffee and future
targets"""

# --- ΜΕΘΟΔΟΙ ΑΝΑΚΑΤΑΣΚΕΥΗΣ ---

def reconstruct_with_langtool(text: str) -> str:
    """Ανακατασκευάζει το κείμενο χρησιμοποιώντας το LanguageTool για γραμματικές διορθώσεις."""
    print("  -> Ανακατασκευή με LanguageTool...")
    tool = language_tool_python.LanguageTool('en-US')
    return tool.correct(text)

def reconstruct_with_spacy(text: str, nlp_model) -> str:
    """Κανονικοποιεί το κείμενο ενώνοντας τις προτάσεις που εντοπίζει το spaCy."""
    print("  -> Ανακατασκευή με spaCy (baseline)...")
    doc = nlp_model(text)
    # Ενώνει τις προτάσεις που βρέθηκαν, αφαιρώντας περιττά κενά και αλλαγές γραμμής.
    sentences = [" ".join(sent.text.split()) for sent in doc.sents]
    return " ".join(sentences)

def reconstruct_with_t5(text: str, model, tokenizer, nlp_model) -> str:
    """Ανακατασκευάζει το κείμενο κάνοντας paraphrase κάθε πρόταση με το T5 model."""
    print("  -> Ανακατασκευή με T5 Paraphrasing (μπορεί να αργήσει)...")
    
    # Χωρίζουμε το κείμενο σε προτάσεις για καλύτερη απόδοση του μοντέλου
    doc = nlp_model(text)
    sentences = [sent.text.strip() for sent in doc.sents if sent.text.strip()]
    
    reconstructed_sentences = []
    
    # Χρησιμοποιούμε το tqdm για να βλέπουμε την πρόοδο
    for sentence in tqdm(sentences, desc="  Paraphrasing sentences"):
        # Για το T5, χρησιμοποιούμε το task prefix για summarization/paraphrasing
        input_text = f"summarize: {sentence}"
        
        # Tokenization με σωστές παραμέτρους
        encoding = tokenizer(
            input_text, 
            return_tensors="pt", 
            max_length=512,  # Αυξημένο όριο για καλύτερη κάλυψη
            truncation=True,
            padding=True
        )
        
        # Δημιουργία της παράφρασης με καλύτερες παραμέτρους
        with torch.no_grad():  # Εξοικονόμηση μνήμης
            outputs = model.generate(
                encoding.input_ids,
                attention_mask=encoding.attention_mask,
                max_length=150,  # Μικρότερο όριο για την έξοδο
                min_length=10,   # Ελάχιστο μήκος
                num_beams=4,
                length_penalty=1.0,
                early_stopping=True,
                do_sample=False
            )
        
        # Αποκωδικοποίηση και καθαρισμός
        reconstructed_sentence = tokenizer.decode(outputs[0], skip_special_tokens=True)
        
        # Αν το μοντέλο επιστρέφει την αρχική είσοδο, χρησιμοποιούμε την ίδια την πρόταση
        if reconstructed_sentence.lower().startswith('summarize:'):
            reconstructed_sentence = sentence
        elif not reconstructed_sentence.strip():
            reconstructed_sentence = sentence
            
        reconstructed_sentences.append(reconstructed_sentence)
        
    return " ".join(reconstructed_sentences)

def reconstruct_custom(text: str) -> str:
    """
    Εφαρμόζει χειροκίνητους, στοχευμένους αυτόματους κανόνες για τη διόρθωση συχνών λαθών.
    Αυτοί οι κανόνες είναι πιο γενικοί από μια απλή αντικατάσταση πρότασης.
    """
    print("  -> Ανακατασκευή με Custom Κανόνες...")
    corrected_text = text

    # Κανόνας 1: Διόρθωση "final discuss" σε "final discussion"
    # Αυτό διορθώνει ένα κοινό λάθος στον δεύτερο κείμενο.
    corrected_text = corrected_text.replace("final discuss", "final discussion")

    # Κανόνας 2: Διόρθωση "Thank your message" σε "Thank you for your message"
    # Αυτό διορθώνει ένα κοινό λάθος στο πρώτο κείμενο.
    # Χρησιμοποιούμε regex για να είμαστε σίγουροι ότι πιάνουμε την αρχή πρότασης ή μετά από τελεία.
    import re
    corrected_text = re.sub(r'(?i)(^|\.\s*)thank your message', r'\1Thank you for your message', corrected_text)

    return corrected_text


# --- ΚΥΡΙΟ ΜΕΡΟΣ ΤΟΥ SCRIPT ---

if __name__ == "__main__":
    print("Ξεκινά η διαδικασία ανακατασκευής κειμένων...")

    # Φόρτωση των μοντέλων μία φορά για εξοικονόμηση χρόνου
    print("Φόρτωση μοντέλων NLP (spaCy, T5)...")
    try:
        nlp_spacy = spacy.load("en_core_web_sm")
        t5_model = T5ForConditionalGeneration.from_pretrained('t5-base')
        t5_tokenizer = T5Tokenizer.from_pretrained('t5-base')
    except Exception as e:
        print(f"Σφάλμα κατά τη φόρτωση των μοντέλων: {e}")
        print("Βεβαιωθείτε ότι έχετε σύνδεση στο internet για την πρώτη εκτέλεση.")
        exit()

    results_data = []
    texts_to_process = {"Κείμενο 1": TEXT_1, "Κείμενο 2": TEXT_2}

    for name, text in texts_to_process.items():
        print(f"\nΕπεξεργασία για το '{name}'...")

        # Εφαρμογή των αυτόματων μεθόδων
        recon_langtool = reconstruct_with_langtool(text)
        recon_spacy = reconstruct_with_spacy(text, nlp_spacy)
        recon_t5 = reconstruct_with_t5(text, t5_model, t5_tokenizer, nlp_spacy)

        # Η custom μέθοδος εφαρμόζεται πλέον σε όλο το κείμενο,
        # καθώς οι κανόνες είναι γενικοί.
        recon_custom_result = reconstruct_custom(text)

        # Αποθήκευση των αποτελεσμάτων
        results_data.append({
            "Original_Text_Name": name,
            "Original_Text": text,
            "Reconstructed_LangTool": recon_langtool,
            "Reconstructed_T5": recon_t5,
            "Reconstructed_Spacy": recon_spacy,
            "Reconstructed_Custom": recon_custom_result
        })


    # Δημιουργία και αποθήκευση του DataFrame
    print("\nΔημιουργία DataFrame και αποθήκευση σε CSV...")
    df = pd.DataFrame(results_data)
    
    # Χρησιμοποιούμε utf-8-sig encoding για να ανοίγει σωστά στο Excel
    df.to_csv("reconstructions.csv", index=False, encoding='utf-8-sig')

    print("\nΗ διαδικασία ολοκληρώθηκε με επιτυχία!")
    print("Τα αποτελέσματα αποθηκεύτηκαν στο αρχείο 'reconstructions.csv'.")