"""
Dataset utilities: EP corpus construction, BM25 index, tokenization helpers.
Ground-truth answers are co-located with query templates for QA evaluation.
"""

import re
import math
import numpy as np
from collections import defaultdict
from config import Config


# ---------------------------------------------------------------------------
# EP synthetic corpus
# ---------------------------------------------------------------------------

EP_CORPUS_DOCUMENTS = [
    # arrhythmia_management  (doc_id 0-9)
    "Supraventricular tachycardia (SVT) management during electrophysiology procedures requires "
    "careful titration of adenosine. The typical dose is 6 mg IV rapid push, followed by 12 mg "
    "if unsuccessful. Vagal maneuvers should be attempted first in stable patients.",

    "Atrial fibrillation (AF) rate control in the perioperative setting commonly uses beta-blockers "
    "such as metoprolol or esmolol. Target resting heart rate is below 110 bpm for lenient control "
    "or below 80 bpm for strict control. Digoxin may be added for refractory cases.",

    "Ventricular tachycardia (VT) storm management involves IV amiodarone loading at 150 mg over "
    "10 minutes followed by 1 mg/min infusion. Catheter ablation is considered for recurrent "
    "monomorphic VT refractory to antiarrhythmic therapy.",

    "Atrioventricular nodal reentrant tachycardia (AVNRT) is the most common form of SVT. "
    "Radiofrequency ablation of the slow pathway achieves cure rates exceeding 95%. "
    "Cryoablation is an alternative with lower risk of AV block.",

    "Atrial flutter (AFL) with typical counterclockwise rotation is treated by ablation of the "
    "cavotricuspid isthmus. Success rates exceed 90%. Anticoagulation management mirrors that of AF.",

    "Wolff-Parkinson-White syndrome (WPW) poses anesthetic risk due to accessory pathway conduction. "
    "Avoid digoxin and verapamil which may accelerate conduction. Procainamide or ibutilide "
    "are preferred for acute AF termination in WPW.",

    "Premature ventricular contractions (PVCs) in structurally normal hearts are generally benign. "
    "High-burden PVCs (>15% of beats) may cause cardiomyopathy. Ablation is effective when "
    "origin is from right ventricular outflow tract.",

    "Sinus node dysfunction (SND) presenting as sick sinus syndrome may require temporary pacing "
    "during anesthesia. Atropine response testing guides pacemaker dependency assessment. "
    "Isoproterenol can provide chronotropic support acutely.",

    "Ventricular fibrillation (VF) requires immediate defibrillation at 200 J biphasic. "
    "CPR should continue until defibrillator is charged. Epinephrine 1 mg IV every 3-5 minutes "
    "and amiodarone 300 mg IV are administered during refractory VF.",

    "Premature atrial contractions (PACs) triggering paroxysmal AF may be treated by pulmonary "
    "vein isolation. Cryoballoon ablation offers single-shot isolation of pulmonary veins. "
    "Success rates at one year approximate 70-80% for paroxysmal AF.",

    # device_indications  (doc_id 10-19)
    "Implantable cardioverter defibrillator (ICD) implantation is indicated for secondary prevention "
    "after survived sudden cardiac death not due to reversible cause. Primary prevention indication "
    "requires LVEF ≤35% with NYHA class II-III symptoms on optimal medical therapy.",

    "Cardiac resynchronization therapy (CRT) is indicated for LVEF ≤35%, LBBB with QRS ≥150 ms, "
    "and NYHA class II-IV symptoms. CRT-D combines defibrillation capability with resynchronization. "
    "Response rate approximates 65-70% with reduction in hospitalizations.",

    "Subcutaneous ICD (S-ICD) avoids transvenous leads and is preferred in young patients or those "
    "with venous access issues. Screening for T-wave oversensing is mandatory pre-implant. "
    "S-ICD cannot provide antibradycardia or antitachycardia pacing.",

    "Leadless pacemakers such as Micra transcatheter pacing system are indicated for patients "
    "with limited venous access or high infection risk. Single-chamber VVI pacing is provided. "
    "Battery longevity exceeds 10 years at nominal settings.",

    "Wearable cardioverter defibrillator (WCD) bridges patients awaiting ICD implantation or "
    "during lead extraction. Minimum wear time of 22 hours per day is recommended. "
    "Compliance monitoring is essential for effective protection.",

    "His bundle pacing preserves physiological conduction and is an alternative to right ventricular "
    "apical pacing. Threshold stability and sensing adequacy require careful intraoperative assessment. "
    "Left bundle branch area pacing offers improved threshold stability.",

    "Dual-chamber pacemakers (DDD) are preferred over single-chamber devices to maintain AV synchrony. "
    "Rate-responsive pacing (DDDR) uses accelerometers or minute ventilation sensors. "
    "Mode switching prevents tracking of atrial tachyarrhythmias.",

    "Epicardial lead systems are used when transvenous access is contraindicated, such as in "
    "congenital heart disease with intracardiac shunts. Steroid-eluting leads reduce threshold rise. "
    "Epicardial thresholds are generally higher than endocardial thresholds.",

    "Generator replacement requires threshold and sensing assessment. Lead impedance trending "
    "identifies insulation failure or conductor fracture. Elective replacement indicator (ERI) "
    "triggers 3-month replacement window.",

    "Remote monitoring of cardiac implantable electronic devices (CIEDs) reduces clinic visits "
    "and enables early detection of arrhythmias and device malfunction. Daily transmission "
    "capability exists in most contemporary devices.",

    # perioperative_anticoagulation  (doc_id 20-29)
    "Perioperative anticoagulation bridging for AF patients with mechanical valves requires "
    "therapeutic low molecular weight heparin. Direct oral anticoagulants (DOACs) do not require "
    "bridging for most AF patients undergoing low-to-intermediate bleeding risk procedures.",

    "Warfarin interruption before surgery: stop 5 days prior for INR normalization. Resume "
    "12-24 hours postoperatively when hemostasis is assured. INR check on day of surgery "
    "is recommended; vitamin K reversal if INR >1.5 for urgent procedures.",

    "DOAC management perioperatively: apixaban and rivaroxaban held 24-48 hours pre-procedure "
    "depending on bleeding risk and renal function. Dabigatran requires 48-96 hour hold "
    "with CrCl <50 mL/min. No routine coagulation monitoring needed.",

    "Heparin reversal with protamine sulfate: 1 mg per 100 units of unfractionated heparin "
    "administered in preceding 2-3 hours. Protamine reactions include hypotension, bradycardia, "
    "and pulmonary hypertension. Have epinephrine and calcium available.",

    "Andexanet alfa reverses factor Xa inhibitors (apixaban, rivaroxaban) for life-threatening "
    "bleeding. Idarucizumab reverses dabigatran. Four-factor PCC is used when specific "
    "reversal agents are unavailable.",

    "Antiplatelet therapy management: aspirin continuation is generally recommended perioperatively "
    "for high cardiovascular risk patients. P2Y12 inhibitors (clopidogrel, ticagrelor) held "
    "5-7 days before procedures with high bleeding risk.",

    "Left atrial appendage occlusion (LAAO) with Watchman device allows anticoagulation "
    "discontinuation in AF patients with high bleeding risk. Post-implant DAPT for 6 months "
    "followed by aspirin monotherapy.",

    "Thromboprophylaxis after EP procedures: femoral venous access sites have low DVT risk. "
    "Early ambulation is encouraged. Prolonged immobility or large sheath sizes increase "
    "thrombotic risk requiring prophylactic anticoagulation.",

    "Heparin-induced thrombocytopenia (HIT) management requires immediate cessation of all "
    "heparin products. Direct thrombin inhibitors (argatroban, bivalirudin) are used for "
    "anticoagulation. Platelet transfusion is contraindicated.",

    "Antiphospholipid syndrome patients require therapeutic anticoagulation with warfarin "
    "targeting INR 2.5-3.5 for arterial thrombosis. DOACs have higher failure rates in "
    "triple-positive antiphospholipid syndrome.",

    # anesthetic_drug_interactions  (doc_id 30-39)
    "Volatile anesthetic agents (sevoflurane, desflurane) prolong QTc interval and may trigger "
    "torsades de pointes in susceptible patients. Baseline QTc assessment and avoidance of "
    "concomitant QT-prolonging drugs is recommended.",

    "Propofol infusion syndrome (PRIS) presents with metabolic acidosis, rhabdomyolysis, and "
    "cardiac failure. Risk increases with doses >4 mg/kg/h for >48 hours. Monitor CK, "
    "triglycerides, and lactate during prolonged infusions.",

    "Succinylcholine causes transient hyperkalemia of 0.5-1.0 mEq/L in normal patients. "
    "Contraindicated in burns, crush injuries, denervation, and prolonged immobility. "
    "Rocuronium with sugammadex reversal is the preferred alternative.",

    "Ketamine increases sympathetic tone, heart rate, and blood pressure. It may be beneficial "
    "in hemodynamically compromised patients. Avoid in patients with severe hypertension "
    "or ischemic heart disease without adequate sympathetic blunting.",

    "Dexmedetomidine causes dose-dependent bradycardia and hypotension via central alpha-2 "
    "agonism. It is useful for sedation during EP mapping procedures requiring patient cooperation. "
    "Loading dose of 1 mcg/kg over 10 minutes followed by 0.2-0.7 mcg/kg/h.",

    "Remifentanil ultra-short-acting opioid is metabolized by plasma esterases. Provides "
    "excellent intraoperative analgesia with rapid offset. Acute opioid tolerance and "
    "hyperalgesia may occur with prolonged infusions.",

    "Neostigmine reversal of neuromuscular blockade increases acetylcholine and may cause "
    "bradycardia. Glycopyrrolate or atropine coadministration prevents bradycardia. "
    "Sugammadex avoids cholinergic side effects.",

    "Ephedrine and phenylephrine are first-line vasopressors for anesthesia-induced hypotension. "
    "Ephedrine has both alpha and beta effects; phenylephrine is pure alpha agonist causing "
    "reflex bradycardia. Vasopressin is used for refractory vasodilatory shock.",

    "Local anesthetic systemic toxicity (LAST) presents with CNS excitation followed by "
    "cardiovascular collapse. Lipid emulsion therapy (intralipid 20%) is the antidote. "
    "Bupivacaine has highest cardiotoxicity; ropivacaine and levobupivacaine are safer.",

    "Magnesium sulfate has antiarrhythmic properties and is used for torsades de pointes "
    "and hypomagnesemia-related arrhythmias. Dose: 2g IV over 15 minutes. "
    "Monitor for hypermagnesemia causing respiratory depression and hypotension.",

    # hemodynamic_emergencies  (doc_id 40-49)
    "Cardiogenic shock during EP procedures requires immediate vasopressor support. "
    "Norepinephrine is first-line for mixed cardiogenic-vasodilatory shock. "
    "Intra-aortic balloon pump or Impella may be required for refractory cases.",

    "Cardiac tamponade complicating catheter ablation presents with hypotension, jugular venous "
    "distension, and muffled heart sounds. Emergent pericardiocentesis is life-saving. "
    "Fluoroscopic or echocardiographic guidance improves safety.",

    "Vasovagal syncope during EP procedures is managed with Trendelenburg positioning, IV fluids, "
    "and atropine for bradycardia. Phenylephrine addresses vasodilation. "
    "Temporary pacing may be required for prolonged asystole.",

    "Anaphylaxis during contrast administration requires epinephrine 0.3-0.5 mg IM immediately. "
    "Diphenhydramine and corticosteroids are adjuncts. Airway management takes priority. "
    "Pre-medication with steroids and antihistamines for known contrast allergy.",

    "Pulmonary embolism during EP procedure presents with sudden hypoxia and hypotension. "
    "Systemic thrombolysis with alteplase 100 mg IV over 2 hours for massive PE. "
    "Catheter-directed thrombolysis or surgical embolectomy for failed systemic therapy.",

    "Aortic dissection as rare complication of retrograde arterial access requires immediate "
    "blood pressure control with esmolol and nitroprusside. CT angiography confirms diagnosis. "
    "Type A dissection requires emergent surgical repair.",

    "Hypertensive emergency during EP procedures: target 20-25% MAP reduction in first hour. "
    "Clevidipine or nicardipine provide titratable IV antihypertensive effect. "
    "Avoid rapid overcorrection causing cerebral hypoperfusion.",

    "Air embolism during central venous access causes mill-wheel murmur and cardiovascular collapse. "
    "Place patient in left lateral decubitus Trendelenburg position. "
    "Aspiration through central venous catheter and 100% oxygen administration.",

    "Malignant hyperthermia triggered by volatile agents or succinylcholine requires "
    "dantrolene 2.5 mg/kg IV as initial dose, repeated every 5 minutes up to 10 mg/kg. "
    "Discontinue triggering agents and provide active cooling.",

    "Hemorrhagic shock from vascular access complications requires immediate volume resuscitation "
    "with crystalloid and blood products in 1:1:1 ratio (PRBC:FFP:platelets). "
    "Damage control surgery or interventional radiology for ongoing hemorrhage.",
]


def expand_abbreviations(text: str, abbrev_map: dict) -> str:
    """Expand medical abbreviations in text."""
    for abbrev, expansion in abbrev_map.items():
        pattern = r'\b' + re.escape(abbrev) + r'\b'
        text = re.sub(pattern, f"{abbrev} ({expansion})", text)
    return text


def build_ep_corpus(config: Config) -> list:
    """Return list of EP corpus document strings with abbreviation expansion."""
    docs = []
    for doc in EP_CORPUS_DOCUMENTS:
        expanded = expand_abbreviations(doc, config.ep_abbreviation_map)
        docs.append(expanded)
    return docs


def chunk_document(text: str, chunk_size: int = 150, overlap: int = 20) -> list:
    """Split document into overlapping word-level chunks."""
    words = text.split()
    chunks = []
    start = 0
    while start < len(words):
        end = min(start + chunk_size, len(words))
        chunk = " ".join(words[start:end])
        chunks.append(chunk)
        if end == len(words):
            break
        start += chunk_size - overlap
    return chunks


def build_corpus_chunks(docs: list, chunk_size: int = 150, overlap: int = 20) -> list:
    """Build flat list of chunks from all documents."""
    all_chunks = []
    for doc_id, doc in enumerate(docs):
        chunks = chunk_document(doc, chunk_size, overlap)
        for chunk_id, chunk in enumerate(chunks):
            all_chunks.append({
                "doc_id": doc_id,
                "chunk_id": chunk_id,
                "text": chunk,
            })
    return all_chunks


# ---------------------------------------------------------------------------
# Clinical query construction WITH ground-truth answers
# ---------------------------------------------------------------------------
# Each entry: (question_string, ground_truth_answer_string)
# Ground truths are extractive spans from EP_CORPUS_DOCUMENTS above.

QUERY_TEMPLATES = {
    "arrhythmia_management": [
        (
            "What is the first-line treatment for SVT during an EP procedure?",
            "adenosine 6 mg IV rapid push with vagal maneuvers attempted first in stable patients",
        ),
        (
            "How should VT storm be managed in the catheterization laboratory?",
            "IV amiodarone loading at 150 mg over 10 minutes followed by 1 mg/min infusion",
        ),
        (
            "What are the ablation success rates for AVNRT?",
            "radiofrequency ablation of the slow pathway achieves cure rates exceeding 95%",
        ),
        (
            "How is atrial flutter treated with catheter ablation?",
            "ablation of the cavotricuspid isthmus with success rates exceeding 90%",
        ),
        (
            "What drugs should be avoided in WPW syndrome?",
            "avoid digoxin and verapamil which may accelerate conduction",
        ),
        (
            "What is the role of amiodarone in VT management?",
            "IV amiodarone loading at 150 mg over 10 minutes followed by 1 mg/min infusion for VT storm",
        ),
        (
            "When is catheter ablation indicated for PVCs?",
            "ablation is effective when origin is from right ventricular outflow tract",
        ),
        (
            "How is sinus node dysfunction managed perioperatively?",
            "temporary pacing during anesthesia with atropine response testing and isoproterenol for chronotropic support",
        ),
        (
            "What is the defibrillation protocol for ventricular fibrillation?",
            "immediate defibrillation at 200 J biphasic with epinephrine 1 mg IV every 3-5 minutes",
        ),
        (
            "What triggers paroxysmal AF from pulmonary veins?",
            "premature atrial contractions triggering paroxysmal AF treated by pulmonary vein isolation",
        ),
    ],
    "device_indications": [
        (
            "What are the primary prevention ICD indications?",
            "LVEF 35% or less with NYHA class II-III symptoms on optimal medical therapy",
        ),
        (
            "When is CRT indicated for heart failure patients?",
            "LVEF 35% or less LBBB with QRS 150 ms or more and NYHA class II-IV symptoms",
        ),
        (
            "What are the advantages of subcutaneous ICD?",
            "avoids transvenous leads preferred in young patients or those with venous access issues",
        ),
        (
            "What is the battery longevity of leadless pacemakers?",
            "battery longevity exceeds 10 years at nominal settings",
        ),
        (
            "How does a wearable cardioverter defibrillator work?",
            "bridges patients awaiting ICD implantation with minimum wear time of 22 hours per day",
        ),
        (
            "What is His bundle pacing and its advantages?",
            "preserves physiological conduction as alternative to right ventricular apical pacing",
        ),
        (
            "When is dual-chamber pacing preferred over single-chamber?",
            "dual-chamber pacemakers preferred to maintain AV synchrony",
        ),
        (
            "What are indications for epicardial lead systems?",
            "when transvenous access is contraindicated such as congenital heart disease with intracardiac shunts",
        ),
        (
            "How is generator replacement managed?",
            "requires threshold and sensing assessment with lead impedance trending and elective replacement indicator",
        ),
        (
            "What are the benefits of remote monitoring for CIEDs?",
            "reduces clinic visits and enables early detection of arrhythmias and device malfunction",
        ),
    ],
    "perioperative_anticoagulation": [
        (
            "Does AF require bridging anticoagulation for low-risk surgery?",
            "DOACs do not require bridging for most AF patients undergoing low-to-intermediate bleeding risk procedures",
        ),
        (
            "How should warfarin be managed before surgery?",
            "stop 5 days prior for INR normalization resume 12-24 hours postoperatively",
        ),
        (
            "When should DOACs be held before an EP procedure?",
            "apixaban and rivaroxaban held 24-48 hours dabigatran requires 48-96 hour hold with CrCl less than 50",
        ),
        (
            "How is heparin reversed with protamine?",
            "1 mg per 100 units of unfractionated heparin administered in preceding 2-3 hours",
        ),
        (
            "What reverses factor Xa inhibitors in emergencies?",
            "andexanet alfa reverses factor Xa inhibitors apixaban rivaroxaban for life-threatening bleeding",
        ),
        (
            "Should aspirin be continued perioperatively?",
            "aspirin continuation is generally recommended perioperatively for high cardiovascular risk patients",
        ),
        (
            "What is the anticoagulation strategy after Watchman implant?",
            "post-implant DAPT for 6 months followed by aspirin monotherapy",
        ),
        (
            "Is thromboprophylaxis needed after EP procedures?",
            "femoral venous access sites have low DVT risk early ambulation is encouraged",
        ),
        (
            "How is heparin-induced thrombocytopenia managed?",
            "immediate cessation of all heparin products with direct thrombin inhibitors argatroban bivalirudin",
        ),
        (
            "What anticoagulation is used in antiphospholipid syndrome?",
            "therapeutic anticoagulation with warfarin targeting INR 2.5-3.5 for arterial thrombosis",
        ),
    ],
    "anesthetic_drug_interactions": [
        (
            "Do volatile anesthetics prolong QTc interval?",
            "volatile anesthetic agents sevoflurane desflurane prolong QTc interval and may trigger torsades de pointes",
        ),
        (
            "What are the risk factors for propofol infusion syndrome?",
            "risk increases with doses greater than 4 mg/kg/h for more than 48 hours",
        ),
        (
            "When is succinylcholine contraindicated?",
            "contraindicated in burns crush injuries denervation and prolonged immobility",
        ),
        (
            "What are the hemodynamic effects of ketamine?",
            "increases sympathetic tone heart rate and blood pressure",
        ),
        (
            "How is dexmedetomidine dosed for EP procedure sedation?",
            "loading dose of 1 mcg/kg over 10 minutes followed by 0.2-0.7 mcg/kg/h",
        ),
        (
            "What causes acute opioid tolerance with remifentanil?",
            "acute opioid tolerance and hyperalgesia may occur with prolonged infusions",
        ),
        (
            "How does neostigmine affect heart rate?",
            "increases acetylcholine and may cause bradycardia requiring glycopyrrolate or atropine",
        ),
        (
            "What vasopressors are used for anesthesia-induced hypotension?",
            "ephedrine and phenylephrine are first-line vasopressors for anesthesia-induced hypotension",
        ),
        (
            "What is the treatment for local anesthetic systemic toxicity?",
            "lipid emulsion therapy intralipid 20% is the antidote",
        ),
        (
            "What is the role of magnesium in arrhythmia management?",
            "magnesium sulfate used for torsades de pointes and hypomagnesemia-related arrhythmias dose 2g IV over 15 minutes",
        ),
    ],
    "hemodynamic_emergencies": [
        (
            "How is cardiogenic shock managed during EP procedures?",
            "norepinephrine is first-line for mixed cardiogenic-vasodilatory shock with intra-aortic balloon pump or Impella",
        ),
        (
            "What are the signs of cardiac tamponade during ablation?",
            "hypotension jugular venous distension and muffled heart sounds requiring emergent pericardiocentesis",
        ),
        (
            "How is vasovagal syncope treated in the EP lab?",
            "Trendelenburg positioning IV fluids and atropine for bradycardia with phenylephrine for vasodilation",
        ),
        (
            "What is the treatment for contrast anaphylaxis?",
            "epinephrine 0.3-0.5 mg IM immediately with diphenhydramine and corticosteroids as adjuncts",
        ),
        (
            "How is massive pulmonary embolism treated?",
            "systemic thrombolysis with alteplase 100 mg IV over 2 hours for massive PE",
        ),
        (
            "What is the management of aortic dissection?",
            "blood pressure control with esmolol and nitroprusside CT angiography type A requires emergent surgical repair",
        ),
        (
            "How should hypertensive emergency be treated during EP?",
            "target 20-25% MAP reduction in first hour with clevidipine or nicardipine",
        ),
        (
            "What is the treatment for air embolism?",
            "left lateral decubitus Trendelenburg position aspiration through central venous catheter and 100% oxygen",
        ),
        (
            "How is malignant hyperthermia treated?",
            "dantrolene 2.5 mg/kg IV as initial dose repeated every 5 minutes up to 10 mg/kg",
        ),
        (
            "How is hemorrhagic shock managed after vascular access complications?",
            "volume resuscitation with crystalloid and blood products in 1:1:1 ratio PRBC FFP platelets",
        ),
    ],
}


def build_clinical_queries(config: Config) -> list:
    """
    Build list of clinical query dicts with category labels and ground-truth answers.
    Each dict has keys: query_id, category, text, original_text, ground_truth.
    """
    queries = []
    for category in config.ep_categories:
        templates = QUERY_TEMPLATES.get(category, [])
        for i, (q_text, gt_answer) in enumerate(
            templates[: config.queries_per_category]
        ):
            expanded = expand_abbreviations(q_text, config.ep_abbreviation_map)
            queries.append(
                {
                    "query_id": len(queries),
                    "category": category,
                    "text": expanded,
                    "original_text": q_text,
                    "ground_truth": gt_answer,
                }
            )
    return queries


# ---------------------------------------------------------------------------
# BM25 implementation (no external dependencies)
# ---------------------------------------------------------------------------


class BM25Index:
    """BM25 retrieval index implemented from scratch."""

    def __init__(self, k1: float = 1.5, b: float = 0.75):
        self.k1 = k1
        self.b = b
        self.corpus = []
        self.tokenized_corpus = []
        self.doc_freqs = []
        self.idf = {}
        self.doc_len = []
        self.avgdl = 0.0
        self.N = 0

    def tokenize(self, text: str) -> list:
        return re.findall(r'\b[a-zA-Z0-9]+\b', text.lower())

    def fit(self, corpus: list):
        """Build BM25 index from list of text strings."""
        self.corpus = corpus
        self.N = len(corpus)
        self.tokenized_corpus = [self.tokenize(doc) for doc in corpus]
        self.doc_len = [len(tokens) for tokens in self.tokenized_corpus]
        self.avgdl = np.mean(self.doc_len) if self.doc_len else 1.0

        # Compute document frequencies
        df = defaultdict(int)
        for tokens in self.tokenized_corpus:
            for term in set(tokens):
                df[term] += 1

        # Compute IDF
        self.idf = {}
        for term, freq in df.items():
            self.idf[term] = math.log(
                (self.N - freq + 0.5) / (freq + 0.5) + 1
            )

        # Term frequencies per document
        self.doc_freqs = []
        for tokens in self.tokenized_corpus:
            tf = defaultdict(int)
            for t in tokens:
                tf[t] += 1
            self.doc_freqs.append(dict(tf))

    def score(self, query: str, doc_idx: int) -> float:
        query_tokens = self.tokenize(query)
        dl = self.doc_len[doc_idx]
        score = 0.0
        tf_map = self.doc_freqs[doc_idx]
        for term in query_tokens:
            if term not in self.idf:
                continue
            tf = tf_map.get(term, 0)
            numerator = tf * (self.k1 + 1)
            denominator = tf + self.k1 * (1 - self.b + self.b * dl / self.avgdl)
            score += self.idf[term] * numerator / denominator
        return score

    def retrieve(self, query: str, top_k: int = 10) -> list:
        """Return list of (doc_idx, score) sorted by score descending."""
        scores = [(i, self.score(query, i)) for i in range(self.N)]
        scores.sort(key=lambda x: x[1], reverse=True)
        return scores[:top_k]


# ---------------------------------------------------------------------------
# TF-IDF dense-like retrieval (deterministic, no external model needed)
# ---------------------------------------------------------------------------


class TFIDFRetriever:
    """TF-IDF based retriever as a deterministic dense retrieval substitute."""

    def __init__(self, max_features: int = 5000):
        self.max_features = max_features
        self.vocab = {}
        self.idf_vec = None
        self.doc_matrix = None
        self.corpus = []

    def tokenize(self, text: str) -> list:
        return re.findall(r'\b[a-zA-Z0-9]+\b', text.lower())

    def fit(self, corpus: list):
        self.corpus = corpus
        tokenized = [self.tokenize(doc) for doc in corpus]

        # Build vocabulary
        df = defaultdict(int)
        for tokens in tokenized:
            for t in set(tokens):
                df[t] += 1

        # Select top-max_features by document frequency
        sorted_terms = sorted(df.items(), key=lambda x: x[1], reverse=True)
        selected = sorted_terms[: self.max_features]
        self.vocab = {term: idx for idx, (term, _) in enumerate(selected)}
        V = len(self.vocab)
        N = len(corpus)

        # IDF
        self.idf_vec = np.zeros(V)
        for term, idx in self.vocab.items():
            self.idf_vec[idx] = math.log((N + 1) / (df[term] + 1)) + 1

        # TF-IDF document matrix
        self.doc_matrix = np.zeros((N, V))
        for d_idx, tokens in enumerate(tokenized):
            tf = defaultdict(int)
            for t in tokens:
                tf[t] += 1
            for t, cnt in tf.items():
                if t in self.vocab:
                    v_idx = self.vocab[t]
                    self.doc_matrix[d_idx, v_idx] = (
                        math.log(cnt + 1) * self.idf_vec[v_idx]
                    )

        # L2 normalize
        norms = np.linalg.norm(self.doc_matrix, axis=1, keepdims=True)
        norms = np.where(norms == 0, 1.0, norms)
        self.doc_matrix = self.doc_matrix / norms

    def query_vector(self, query: str) -> np.ndarray:
        tokens = self.tokenize(query)
        V = len(self.vocab)
        vec = np.zeros(V)
        tf = defaultdict(int)
        for t in tokens:
            tf[t] += 1
        for t, cnt in tf.items():
            if t in self.vocab:
                v_idx = self.vocab[t]
                vec[v_idx] = math.log(cnt + 1) * self.idf_vec[v_idx]
        norm = np.linalg.norm(vec)
        if norm > 0:
            vec = vec / norm
        return vec

    def retrieve(self, query: str, top_k: int = 10) -> list:
        q_vec = self.query_vector(query)
        scores = self.doc_matrix @ q_vec
        top_indices = np.argsort(scores)[::-1][:top_k]
        return [(int(i), float(scores[i])) for i in top_indices]


# ---------------------------------------------------------------------------
# Ground-truth relevance labels (based on category-to-doc mapping)
# ---------------------------------------------------------------------------

# Documents 0-9: arrhythmia_management
# Documents 10-19: device_indications
# Documents 20-29: perioperative_anticoagulation
# Documents 30-39: anesthetic_drug_interactions
# Documents 40-49: hemodynamic_emergencies

CATEGORY_DOC_RANGES = {
    "arrhythmia_management": (0, 10),
    "device_indications": (10, 20),
    "perioperative_anticoagulation": (20, 30),
    "anesthetic_drug_interactions": (30, 40),
    "hemodynamic_emergencies": (40, 50),
}


def get_relevant_chunks(query: dict, all_chunks: list) -> set:
    """Return set of chunk indices relevant to a query based on document category."""
    category = query["category"]
    doc_start, doc_end = CATEGORY_DOC_RANGES[category]
    relevant = set()
    for i, chunk in enumerate(all_chunks):
        if doc_start <= chunk["doc_id"] < doc_end:
            relevant.add(i)
    return relevant