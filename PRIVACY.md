# Privacy & Data Ethics

Privacy is a **design constraint** in EveryCam, not a feature flag. Everyday-camera video is exactly the kind of
data that can harm people if handled carelessly, so EveryCam is built to make the responsible path the default
path.

## Core principles

1. **Anonymize before storage.** Every frame passes through the privacy gate *before* perception and *before*
   anything is written to disk. Faces and license plates are blurred first; the un-blurred frame is never saved.
2. **Fail closed.** If a detector errors, the gate blurs the **entire** frame rather than risk leaking identity.
3. **No identity, ever.** EveryCam has **no** face-recognition, person re-identification, or ID-matching
   capability, and it will not gain one. It extracts *hands, objects, motion, and affordances* — not *who* a
   person is.
4. **Provenance travels with the data.** Each dataset includes `meta/everycam.json` recording source type,
   consent, license, and the fact that anonymization ran.

## What EveryCam will **not** do

EveryCam explicitly does **not**, and will not, support:

- Linking footage to **government identity** (passports, visas, national IDs such as Aadhaar) or any identity
  database.
- Face recognition, facial matching, biometric identification, gait/person re-identification, or individual
  tracking.
- Covert surveillance, or building profiles of identifiable people.

These are out of scope by design. Pull requests adding such capabilities will be declined.

## Responsible-use guidance

You are responsible for the footage you capture. Before using EveryCam on real video:

- **Capture only what you have the right to.** Your own webcam/phone footage, footage of consenting participants,
  or properly licensed / public-domain / synthetic data.
- **Fixed / CCTV-style cameras:** use only consented, lawful, or synthetic feeds. Anonymization reduces — but
  does not eliminate — privacy risk; context (location, who is filmed) matters.
- **Comply with local law** — e.g. GDPR (EU), BIPA (Illinois), the DPDP Act (India), and any jurisdiction-specific
  rules on recording people. EveryCam is a tool, not legal advice.
- **Minimize and delete.** Keep only the derived signals you need; don't hoard raw video.

## Reporting

If you find a privacy weakness (e.g. a case where the gate misses a face), please open a **security/privacy issue**
so it can be fixed. Improving the anonymizer is always a welcome contribution.
