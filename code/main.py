I'll implement a complete, self-contained hybrid RAG benchmarking system for electrophysiology anesthesia clinical decision support. The implementation uses lightweight CPU-friendly algorithms with realistic mock distributions grounded in the research methodology.

```filename:ep_knowledge_base.py
"""
Electrophysiology Anesthesia Knowledge Base
50 clinical chunks covering all taxonomy categories
"""

EP_KNOWLEDGE_BASE = [
    # ============================================================
    # CATEGORY 1: ARRHYTHMIA MANAGEMENT (10 chunks)
    # ============================================================
    {
        "id": "arr_001",
        "category": "arrhythmia_management",
        "title": "SVT Management in EP Lab",
        "content": (
            "Supraventricular tachycardia (SVT) management during EP procedures: "
            "Adenosine 6-12mg IV rapid push is first-line for acute SVT termination. "
            "For refractory SVT, consider synchronized cardioversion at 50-100J biphasic. "
            "Vagal maneuvers (Valsalva, carotid sinus massage) may terminate AVNRT. "
            "Beta-blockers (metoprolol 5mg IV) or calcium channel blockers (verapamil 5-10mg IV) "
            "are second-line agents. Avoid adenosine in pre-excitation syndromes (WPW) with AF "
            "as it may accelerate conduction via accessory pathway causing VF. "
            "During ablation procedures, maintain adequate anticoagulation (ACT >300s)."
        ),
        "source": "ACC/AHA SVT Guidelines 2015",
        "evidence_level": "Class I"
    },
    {
        "id": "arr_002",
        "category": "arrhythmia_management",
        "title": "VT Storm Management",
        "content": (
            "Ventricular tachycardia (VT) storm definition: ≥3 separate VT episodes within 24h "
            "requiring intervention. Management: IV amiodarone 150mg over 10min then 1mg/min infusion. "
            "Lidocaine 1-1.5mg/kg IV bolus as alternative. Beta-blockade with propranolol 0.15mg/kg IV "
            "reduces sympathetic drive. Sedation/anesthesia reduces catecholamine surge. "
            "Consider deep sedation with propofol infusion 25-75mcg/kg/min. "
            "Stellate ganglion block (left) may reduce VT burden in refractory cases. "
            "Urgent catheter ablation indicated if medical therapy fails. "
            "Hemodynamically unstable VT: immediate unsynchronized defibrillation 200J biphasic."
        ),
        "source": "ACC/AHA VT Guidelines 2017",
        "evidence_level": "Class I"
    },
    {
        "id": "arr_003",
        "category": "arrhythmia_management",
        "title": "AF Ablation Anesthesia Protocol",
        "content": (
            "Atrial fibrillation (AF) ablation anesthesia: General anesthesia preferred over "
            "conscious sedation for pulmonary vein isolation (PVI) to minimize patient movement. "
            "Propofol-based TIVA (total intravenous anesthesia) is standard. "
            "Avoid volatile agents that may affect electrophysiology mapping. "
            "Target MAP >65mmHg throughout procedure. Esophageal temperature monitoring mandatory "
            "during posterior wall ablation to prevent atrio-esophageal fistula (keep <38°C). "
            "Heparin anticoagulation: bolus 100 units/kg, maintain ACT 300-350s. "
            "Protamine reversal at procedure end: 1mg per 100 units heparin given."
        ),
        "source": "HRS Expert Consensus AF Ablation 2017",
        "evidence_level": "Class I"
    },
    {
        "id": "arr_004",
        "category": "arrhythmia_management",
        "title": "PVC Ablation Considerations",
        "content": (
            "Premature ventricular contractions (PVC) ablation: Anesthesia consideration is "
            "suppression of PVCs by anesthetic agents. Propofol may suppress PVC burden. "
            "Dexmedetomidine infusion (0.5-1 mcg/kg/h) preferred as it minimally affects PVC frequency. "
            "Avoid deep propofol anesthesia during mapping phase. "
            "Light sedation with midazolam/fentanyl during mapping, deeper for ablation. "
            "Isoproterenol 1-3 mcg/min IV used to provoke PVCs during electrophysiology study. "
            "Epinephrine challenge may unmask RVOT PVCs. "
            "Post-ablation: monitor for 30min for recurrence before recovery."
        ),
        "source": "ACC EP Lab Practice Standards 2019",
        "evidence_level": "Class IIa"
    },
    {
        "id": "arr_005",
        "category": "arrhythmia_management",
        "title": "Bradyarrhythmia Management",
        "content": (
            "Bradyarrhythmia in EP lab: Symptomatic bradycardia during ablation may indicate "
            "inadvertent AV node injury. Immediate actions: stop RF energy, assess rhythm. "
            "Atropine 0.5-1mg IV for vagally-mediated bradycardia. "
            "Transcutaneous pacing pads should be applied before all EP procedures. "
            "Temporary transvenous pacing via femoral or internal jugular approach for "
            "complete heart block. Isoproterenol 2-10 mcg/min for chronotropic support. "
            "Dopamine 2-5 mcg/kg/min for combined chronotropic/inotropic support. "
            "Complete AV block post-ablation: observe 30-60min; if persistent, permanent pacemaker implant."
        ),
        "source": "ACC/AHA Bradyarrhythmia Guidelines 2018",
        "evidence_level": "Class I"
    },
    {
        "id": "arr_006",
        "category": "arrhythmia_management",
        "title": "Torsades de Pointes Management",
        "content": (
            "Torsades de Pointes (TdP) in EP lab: Polymorphic VT associated with QT prolongation. "
            "Immediate: magnesium sulfate 2g IV over 5-10min (first-line regardless of Mg level). "
            "Increase heart rate to shorten QT: isoproterenol infusion or temporary pacing at 90-110 bpm. "
            "Discontinue QT-prolonging drugs (amiodarone, sotalol, haloperidol, ondansetron). "
            "Correct electrolytes: K+ >4.5 mEq/L, Mg2+ >2 mEq/L. "
            "Avoid Class Ia and III antiarrhythmics. Lidocaine (Class Ib) is safe. "
            "Congenital LQTS: beta-blockers, avoid sympathetic stimulation, consider left cardiac "
            "sympathetic denervation. Defibrillate if hemodynamically unstable."
        ),
        "source": "AHA ACLS Guidelines 2020",
        "evidence_level": "Class I"
    },
    {
        "id": "arr_007",
        "category": "arrhythmia_management",
        "title": "Wolff-Parkinson-White Syndrome",
        "content": (
            "WPW syndrome anesthesia: Pre-excitation via accessory pathway poses anesthetic risks. "
            "Avoid digoxin and verapamil (may accelerate accessory pathway conduction). "
            "Adenosine contraindicated if AF present with WPW (risk of VF). "
            "Procainamide 15-17mg/kg IV preferred for AF with WPW. "
            "Induction agents: propofol and fentanyl are safe. "
            "Volatile anesthetics generally safe but may affect conduction properties. "
            "Succinylcholine: use caution, fasciculations may trigger tachyarrhythmia. "
            "Defibrillator immediately available. Post-ablation: delta wave abolition confirms success."
        ),
        "source": "ACC/AHA SVT Guidelines 2015",
        "evidence_level": "Class I"
    },
    {
        "id": "arr_008",
        "category": "arrhythmia_management",
        "title": "Ventricular Fibrillation Protocol",
        "content": (
            "Ventricular fibrillation (VF) in EP lab: Immediate defibrillation 200J biphasic. "
            "If no ROSC: CPR 2 minutes, repeat shock 200-360J. "
            "Epinephrine 1mg IV every 3-5min after second shock. "
            "Amiodarone 300mg IV bolus for refractory VF. "
            "Identify and treat reversible causes (4H's and 4T's). "
            "EP lab-specific considerations: ensure fluoroscopy table allows adequate CPR. "
            "Remove intracardiac catheters if possible during CPR. "
            "Post-ROSC: targeted temperature management if comatose (32-36°C for 24h). "
            "ICD implantation indicated for VF survivors without reversible cause."
        ),
        "source": "AHA ACLS Guidelines 2020",
        "evidence_level": "Class I"
    },
    {
        "id": "arr_009",
        "category": "arrhythmia_management",
        "title": "Atrial Flutter Ablation",
        "content": (
            "Typical atrial flutter (cavotricuspid isthmus-dependent): "
            "Catheter ablation is first-line therapy (Class I indication). "
            "Anesthesia: moderate sedation or general anesthesia. "
            "Anticoagulation: same as AF ablation protocol. "
            "Success rate >95% for CTI-dependent flutter. "
            "Post-ablation: bidirectional conduction block confirmed. "
            "Risk of AV block <1% for typical flutter ablation. "
            "Atypical flutter: requires 3D mapping system, longer procedure time. "
            "Post-procedure monitoring: 4-6h observation, ECG documentation."
        ),
        "source": "HRS Expert Consensus 2017",
        "evidence_level": "Class I"
    },
    {
        "id": "arr_010",
        "category": "arrhythmia_management",
        "title": "AVNRT Ablation Protocol",
        "content": (
            "AV nodal reentrant tachycardia (AVNRT) ablation: "
            "Slow pathway modification is standard approach. "
            "Anesthesia: moderate sedation preferred to allow patient cooperation. "
            "Isoproterenol 1-3 mcg/min for induction of AVNRT during EP study. "
            "Ablation near compact AV node: risk of complete heart block 0.5-1%. "
            "Junctional rhythm during ablation indicates proximity to AV node - reduce power. "
            "Success rate >97%. Recurrence rate 2-5%. "
            "Post-procedure: ECG, 4h monitoring, no anticoagulation required for typical AVNRT."
        ),
        "source": "ACC EP Standards 2019",
        "evidence_level": "Class I"
    },

    # ============================================================
    # CATEGORY 2: DEVICE INDICATIONS (10 chunks)
    # ============================================================
    {
        "id": "dev_001",
        "category": "device_indications",
        "title": "ICD Primary Prevention Indications",
        "content": (
            "Implantable cardioverter-defibrillator (ICD) primary prevention: "
            "Class I indications: LVEF ≤35% with NYHA Class II-III heart failure on optimal medical therapy "
            "for ≥3 months, expected meaningful survival >1 year. "
            "Ischemic cardiomyopathy: LVEF ≤35%, >40 days post-MI, NYHA II-III. "
            "Non-ischemic cardiomyopathy: LVEF ≤35%, NYHA II-III. "
            "ARVC with significant ventricular dysfunction or inducible VT. "
            "Channelopathies: Brugada syndrome with spontaneous Type 1 pattern and syncope. "
            "LQTS with prior cardiac arrest or syncope on beta-blocker therapy. "
            "Anesthesia pre-ICD: optimize heart failure, assess anticoagulation status."
        ),
        "source": "ACC/AHA ICD Guidelines 2018",
        "evidence_level": "Class I"
    },
    {
        "id": "dev_002",
        "category": "device_indications",
        "title": "ICD Secondary Prevention",
        "content": (
            "ICD secondary prevention indications: "
            "Class I: Cardiac arrest due to VF or hemodynamically unstable VT without reversible cause. "
            "Spontaneous sustained VT with structural heart disease. "
            "Syncope with inducible hemodynamically significant VT/VF at EPS. "
            "LVEF ≤40% with nonsustained VT and inducible VF/VT at EPS. "
            "Exclusions: VT/VF within 48h of acute MI (reversible), terminal illness <1 year, "
            "incessant VT/VF (bridge to ablation), NYHA Class IV not candidate for CRT/transplant. "
            "Pre-procedure: assess LVEF, document prior arrhythmia, optimize medications."
        ),
        "source": "ACC/AHA ICD Guidelines 2018",
        "evidence_level": "Class I"
    },
    {
        "id": "dev_003",
        "category": "device_indications",
        "title": "CRT Indications and Programming",
        "content": (
            "Cardiac resynchronization therapy (CRT) indications: "
            "Class I: LBBB with QRS ≥150ms, LVEF ≤35%, NYHA II-IV on optimal medical therapy. "
            "Class IIa: Non-LBBB with QRS ≥150ms, LVEF ≤35%, NYHA III-IV. "
            "LBBB with QRS 120-149ms, LVEF ≤35%, NYHA II-IV. "
            "CRT-D preferred over CRT-P if ICD indication exists. "
            "Anesthesia for CRT: general anesthesia or deep sedation, "
            "maintain hemodynamics during LV lead placement (most challenging step). "
            "Phrenic nerve stimulation testing required post-implant. "
            "Optimization: echocardiographic AV and VV interval optimization at 6 weeks."
        ),
        "source": "ACC/AHA Heart Failure Guidelines 2022",
        "evidence_level": "Class I"
    },
    {
        "id": "dev_004",
        "category": "device_indications",
        "title": "Pacemaker Indications",
        "content": (
            "Permanent pacemaker indications: "
            "Class I: Symptomatic sinus node dysfunction (sick sinus syndrome). "
            "Third-degree (complete) AV block regardless of symptoms. "
            "Second-degree AV block Type II (Mobitz II) regardless of symptoms. "
            "Symptomatic second-degree AV block Type I (Mobitz I). "
            "Bifascicular block with syncope and inducible infra-Hisian block at EPS. "
            "Pacemaker dependency post-cardiac surgery or ablation. "
            "Anesthesia: local with sedation standard; general for uncooperative patients. "
            "Avoid succinylcholine in pacemaker-dependent patients (fasciculations may inhibit pacemaker)."
        ),
        "source": "ACC/AHA Pacemaker Guidelines 2018",
        "evidence_level": "Class I"
    },
    {
        "id": "dev_005",
        "category": "device_indications",
        "title": "Subcutaneous ICD (S-ICD)",
        "content": (
            "Subcutaneous ICD (S-ICD) indications and considerations: "
            "Appropriate when pacing not required (no bradycardia, CRT, or ATP indication). "
            "Preferred in: young patients, those with venous access issues, high infection risk. "
            "Contraindicated if: ATP needed for VT, CRT indication, pacing-dependent. "
            "Screening ECG required to ensure adequate R-wave sensing (SMART Pass algorithm). "
            "Cannot terminate VT with ATP (only shock therapy). "
            "Anesthesia: general anesthesia required for implantation due to pain. "
            "Defibrillation testing at implant: VF induction, confirm 65J shock terminates VF. "
            "Post-implant: 6-week wound check, programming optimization."
        ),
        "source": "ACC/AHA ICD Guidelines 2018",
        "evidence_level": "Class IIa"
    },
    {
        "id": "dev_006",
        "category": "device_indications",
        "title": "Lead Extraction Anesthesia",
        "content": (
            "Cardiac device lead extraction: High-risk procedure requiring cardiac surgery backup. "
            "Indications: device infection, lead malfunction, venous occlusion, upgrade. "
            "Anesthesia: general anesthesia mandatory. "
            "Arterial line, large-bore IV access, type and cross for 4 units pRBC. "
            "TEE monitoring for pericardial effusion and tamponade detection. "
            "Cardiac surgery on standby in OR or hybrid suite. "
            "Major complications: SVC tear, cardiac perforation, tamponade (1-2%). "
            "Vasopressor support: norepinephrine infusion ready. "
            "Post-extraction: ICU admission for high-risk cases, echo at 24h."
        ),
        "source": "HRS Lead Management Consensus 2017",
        "evidence_level": "Class I"
    },
    {
        "id": "dev_007",
        "category": "device_indications",
        "title": "Wearable Cardioverter Defibrillator",
        "content": (
            "Wearable cardioverter defibrillator (WCD) indications: "
            "Bridge to ICD decision in: newly diagnosed cardiomyopathy (LVEF ≤35%), "
            "post-MI waiting period (40 days), post-revascularization (90 days), "
            "device explant for infection. "
            "WCD delivers 150J shock for VT/VF detection. "
            "Anesthesia consideration: WCD must be removed before MRI. "
            "During surgery: remove WCD, apply external defibrillator pads. "
            "Reassess LVEF at 3 months to determine permanent ICD need. "
            "Compliance monitoring via remote transmission. "
            "Shock delivery: patient can abort inappropriate shock by pressing button."
        ),
        "source": "ACC/AHA ICD Guidelines 2018",
        "evidence_level": "Class IIb"
    },
    {
        "id": "dev_008",
        "category": "device_indications",
        "title": "CIED Management Perioperative",
        "content": (
            "Cardiovascular implantable electronic device (CIED) perioperative management: "
            "Pre-operative: device interrogation within 6 months (12 months if stable). "
            "Determine pacing dependency. "
            "Reprogramming indications: pacemaker-dependent patient undergoing surgery with "
            "electrocautery above umbilicus - reprogram to asynchronous mode (DOO/VOO). "
            "Magnet application: converts most pacemakers to asynchronous, most ICDs to monitor-only. "
            "Deactivate ICD therapies for surgery (electrocautery may trigger inappropriate shock). "
            "Post-operative: restore original settings, re-enable ICD therapies. "
            "Remote monitoring setup within 24h post-procedure."
        ),
        "source": "HRS CIED Perioperative Consensus 2011",
        "evidence_level": "Class I"
    },
    {
        "id": "dev_009",
        "category": "device_indications",
        "title": "Conduction System Pacing",
        "content": (
            "Conduction system pacing (His bundle and left bundle branch pacing): "
            "Indications: AV block requiring pacing to avoid RV pacing-induced cardiomyopathy, "
            "CRT upgrade in non-responders, pacing-induced cardiomyopathy. "
            "His bundle pacing: more physiologic, maintains narrow QRS. "
            "LBBP: easier implant, more stable sensing, higher success rate than HBP. "
            "Anesthesia: same as standard pacemaker implant (local + sedation). "
            "Longer fluoroscopy time expected (learning curve). "
            "Threshold optimization: target His capture threshold <2V at 1ms. "
            "Post-implant: 12-lead ECG to confirm narrow QRS and appropriate capture."
        ),
        "source": "HRS Expert Consensus 2021",
        "evidence_level": "Class IIa"
    },
    {
        "id": "dev_010",
        "category": "device_indications",
        "title": "ICD Shock Management",
        "content": (
            "ICD shock management in EP lab: "
            "Appropriate shock: confirmed VT/VF episode, device functioned correctly. "
            "Inappropriate shock: SVT, T-wave oversensing, lead fracture, EMI. "
            "Post-shock evaluation: 12-lead ECG, device interrogation, cardiac biomarkers. "
            "Anesthesia for ICD testing (DFT): brief general anesthesia or deep sedation. "
            "Propofol 1-2mg/kg for induction, maintain with propofol infusion. "
            "Ensure external defibrillator available and charged. "
            "Minimize time in VF (<10s before backup shock if ICD fails). "
            "Post-DFT: monitor 2h, document sensing thresholds and shock impedance."
        ),
        "source": "ACC EP Standards 2019",
        "evidence_level": "Class I"
    },

    # ============================================================
    # CATEGORY 3: PERIOPERATIVE ANTICOAGULATION (10 chunks)
    # ============================================================
    {
        "id": "anti_001",
        "category": "perioperative_anticoagulation",
        "title": "Heparin Protocol AF Ablation",
        "content": (
            "Heparin anticoagulation for AF ablation: "
            "Unfractionated heparin (UFH) bolus: 100-150 units/kg IV at transseptal puncture. "
            "Target ACT: 300-350 seconds throughout procedure. "
            "Check ACT every 15-30 minutes. "
            "Additional heparin boluses: 1000-2000 units if ACT <300s. "
            "Continuous infusion: 1000 units/hour as alternative. "
            "Reversal: protamine sulfate 1mg per 100 units heparin remaining. "
            "Uninterrupted anticoagulation strategy: continue warfarin (INR 2-3) or DOAC through ablation. "
            "Advantages: reduced thromboembolic events, no heparin bridging needed. "
            "Post-ablation: resume DOAC 3-4h after hemostasis achieved."
        ),
        "source": "HRS AF Ablation Consensus 2017",
        "evidence_level": "Class I"
    },
    {
        "id": "anti_002",
        "category": "perioperative_anticoagulation",
        "title": "DOAC Management Perioperative",
        "content": (
            "Direct oral anticoagulant (DOAC) perioperative management in EP: "
            "Apixaban/Rivaroxaban: hold 24h before low-bleed-risk procedures, 48h for high-risk. "
            "Dabigatran: hold 24-48h (CrCl >50), 48-72h (CrCl 30-50). "
            "No bridging required for DOACs (unlike warfarin). "
            "Resume DOAC: 6-8h post-procedure for low-bleed-risk, 48-72h for high-risk. "
            "Reversal agents: idarucizumab for dabigatran (5g IV), "
            "andexanet alfa for factor Xa inhibitors. "
            "Emergency reversal: 4-factor PCC (50 units/kg) if reversal agent unavailable. "
            "AF ablation: uninterrupted DOAC preferred (AXAFA-AFNET 5 trial evidence)."
        ),
        "source": "ACC/AHA Anticoagulation Guidelines 2019",
        "evidence_level": "Class I"
    },
    {
        "id": "anti_003",
        "category": "perioperative_anticoagulation",
        "title": "Warfarin Management EP Procedures",
        "content": (
            "Warfarin management for EP procedures: "
            "AF ablation: continue warfarin with therapeutic INR (2-3) - uninterrupted strategy. "
            "Device implantation (pacemaker/ICD): continue warfarin with INR ≤3.5 preferred. "
            "BRUISE CONTROL trial: continued warfarin superior to heparin bridging for device implant "
            "(reduced pocket hematoma: 3.5% vs 16%). "
            "Bridging anticoagulation: only for mechanical heart valves or very high stroke risk. "
            "Bridging with LMWH: enoxaparin 1mg/kg SQ BID, last dose 24h pre-procedure. "
            "Post-procedure warfarin restart: evening of procedure if hemostasis adequate. "
            "INR check at 3-5 days post-restart."
        ),
        "source": "BRUISE CONTROL Trial; ACC Guidelines 2017",
        "evidence_level": "Class I"
    },
    {
        "id": "anti_004",
        "category": "perioperative_anticoagulation",
        "title": "Bleeding Risk Stratification",
        "content": (
            "Bleeding risk stratification for EP procedures: "
            "Low bleeding risk: pacemaker/ICD implant, EP study without ablation, cardioversion. "
            "High bleeding risk: complex AF ablation (posterior wall, roof lines), "
            "epicardial access, lead extraction, VT ablation with epicardial approach. "
            "HAS-BLED score for warfarin patients: score ≥3 indicates high bleeding risk. "
            "Thrombocytopenia: platelet count <50,000 - transfuse before procedure. "
            "Platelet dysfunction: hold aspirin 5-7 days, clopidogrel 5-7 days pre-procedure. "
            "Dual antiplatelet: DAPT continuation acceptable for most EP procedures. "
            "Protamine allergy: pretreatment with diphenhydramine/steroids, have epinephrine ready."
        ),
        "source": "ACC/AHA Perioperative Guidelines 2014",
        "evidence_level": "Class IIa"
    },
    {
        "id": "anti_005",
        "category": "perioperative_anticoagulation",
        "title": "Tamponade Management and Anticoagulation",
        "content": (
            "Cardiac tamponade during EP procedure: "
            "Recognition: hypotension, tachycardia, equalization of pressures, echo confirmation. "
            "Immediate: stop anticoagulation, call cardiac surgery. "
            "Pericardiocentesis: subxiphoid approach under echo/fluoro guidance. "
            "Heparin reversal: protamine 1mg per 100 units heparin (given over 10min). "
            "Autotransfusion: blood from pericardial drain can be reinfused. "
            "If hemodynamics deteriorate: emergent surgical drainage. "
            "Post-tamponade anticoagulation: restart cautiously after 24-48h with echo monitoring. "
            "Incidence: AF ablation 0.5-1%, VT ablation 1-2%, lead extraction 1-2%."
        ),
        "source": "HRS Expert Consensus Complications 2011",
        "evidence_level": "Class I"
    },
    {
        "id": "anti_006",
        "category": "perioperative_anticoagulation",
        "title": "Anticoagulation Reversal Emergency",
        "content": (
            "Emergency anticoagulation reversal in EP lab: "
            "UFH reversal: protamine sulfate, max 50mg IV over 10min. Excess protamine is anticoagulant. "
            "Warfarin reversal: 4-factor PCC (KCentra) 25-50 units/kg + Vitamin K 10mg IV. "
            "Dabigatran: idarucizumab (Praxbind) 5g IV (2 x 2.5g vials). "
            "Apixaban/rivaroxaban: andexanet alfa (Andexxa) - low dose: 400mg bolus + 480mg infusion; "
            "high dose: 800mg bolus + 960mg infusion. "
            "If reversal agents unavailable: 4-factor PCC 50 units/kg. "
            "Tranexamic acid 1g IV for fibrinolytic state. "
            "Maintain product availability: type and cross, FFP, platelets."
        ),
        "source": "AHA Bleeding Management Statement 2020",
        "evidence_level": "Class I"
    },
    {
        "id": "anti_007",
        "category": "perioperative_anticoagulation",
        "title": "Stroke Prevention Post-Ablation",
        "content": (
            "Stroke prevention after AF ablation: "
            "CHA2DS2-VASc score guides long-term anticoagulation decision. "
            "Score ≥2 (men) or ≥3 (women): continue anticoagulation indefinitely regardless of ablation success. "
            "Score 0-1: anticoagulation may be discontinued if confirmed AF-free at 3 months. "
            "Peri-ablation: uninterrupted anticoagulation reduces stroke risk. "
            "Post-ablation anticoagulation: minimum 2 months regardless of rhythm. "
            "Cardiac CT/TEE: rule out LAA thrombus before ablation if DOAC held >48h. "
            "Cerebral microemboli: more common with dry RF ablation vs. irrigated-tip. "
            "Silent cerebral infarction incidence: 5-15% post-AF ablation on DWI MRI."
        ),
        "source": "HRS/EHRA/ECAS AF Ablation Consensus 2017",
        "evidence_level": "Class I"
    },
    {
        "id": "anti_008",
        "category": "perioperative_anticoagulation",
        "title": "Antiplatelet Therapy EP Procedures",
        "content": (
            "Antiplatelet therapy management for EP procedures: "
            "Aspirin: generally continue for all EP procedures (low bleeding risk). "
            "Clopidogrel: continue for low-risk procedures; hold 5-7 days for high-risk. "
            "Ticagrelor: hold 5-7 days before high-