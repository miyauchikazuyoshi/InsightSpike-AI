# Maker Sovereignty for Digital Artifacts  
## A Semi‑Decentralized Infrastructure for Ownership and Safety

_Reclaiming Blockchain as Public Registry, Not Speculation_  
Author: Kazuyoshi Miyauchi (Independent Researcher / InsightSpike‑AI)  
Date: November 2025  
Version: v4.0 — Integrated Draft (English rewrite)

> This file is an English markdown version of the Japanese proposal  
> `maker_sovereignty_full_proposal (1).pdf`. It preserves the structure  
> and main arguments while simplifying some wording for clarity.

---

## Executive Summary

This proposal presents a semi‑decentralized licensing and registry infrastructure that aims to resolve three intertwined crises of the digital era:

1. **Loss of meaningful ownership**  
   - Subscription dominance and cloud dependence have eroded the very notion of “buying” digital goods.  
   - Users cannot resell, gift, or bequeath many digital assets; platform shutdowns can wipe out entire libraries.
2. **Safety risks from AI and OSS capability externalization**  
   - Powerful models and tools can be copied and recombined cheaply, enabling severe human‑rights violations.  
   - Classical open‑source assumptions (“externalization is costly, transparency is always net‑positive”) are breaking.
3. **The financialization of blockchain**  
   - Blockchains have drifted from their original role as tamper‑resistant public registries  
     toward speculative tokens, ICOs, and casino‑like DeFi, undermining their civic potential.

To address these, the proposal argues for a layered design:

- **Technical layer:** _Approve → Token → Key → Execute_  
  - Hardware‑backed identity (TPM/TEE), cryptographic keys, and revocation lists (CRL) gate access to sensitive capabilities.  
  - Distribution and execution are mediated by signed tokens, not raw files.
- **Economic layer:** _Resale royalties + cost sharing_  
  - True resale, gifting, and secondary markets are restored for digital goods.  
  - Sustainable incentives are built in: the more safely and widely a work is reused, the more the maker benefits.
- **Ideational / institutional layer:** _“Administrative blockchain”_  
  - Blockchains are repositioned as neutral public registries, analogous to land registries, mints, or notary systems.  
  - Instead of permissionless casinos or fully centralized silos, a semi‑decentralized registry anchors provenance and rights.

The target outcome is **maker sovereignty**:

- Makers retain enough control to be _responsible_ for what they create.  
- At the same time, no single private actor can permanently monopolize execution or distribution.

The system is designed for three main domains:

1. **VR / games** — ownership‑centric (gifting, resale, “anti‑1984” for virtual worlds).  
2. **Commercial AI** — safety‑centric (revocation rights, genealogy tracking of dangerous capabilities).  
3. **OSS research** — both (parallel distribution, dual‑key model for open experimentation and controlled deployment).

The core insights can be summarized as:

1. **Blockchain ≠ speculative asset.** Its natural role is a public registry of facts and commitments.  
2. **Ownership and safety are not in conflict.** When resale, royalties, and revocation are wired into the same
   infrastructure, safety becomes part of the same feedback loop that sustains creators economically.  
3. **Neither pure centralization nor pure decentralization is viable.** A carefully governed semi‑decentralized layer is the practical optimum.

---

## Part I — Problems and Motivation (Why)

### 1. Crisis 1: Loss of Ownership in the Digital Era

#### 1.1 Current Situation

Digital distribution has shifted the default from ownership to subscription and platform‑dependence:

- **Subscription dependence:**  
  - Software suites (e.g., creative tools) are available only as ongoing subscriptions.  
  - Stopping payment often means instant loss of access—even if past usage would justify a perpetual license.
- **Cloud concentration:**  
  - Assets “live” on vendor servers; if the provider shuts down or bans an account, entire libraries can vanish overnight.
- **No resale, no gifting:**  
  - Game libraries, ebooks, and digital media frequently cannot be resold, gifted, or inherited.  
  - There is no equivalent of used bookstores or second‑hand game shops.
- **Psychological impact:**  
  - People hesitate to “buy” digital content they cannot control, resell, or pass on, leading to weaker cultural circulation.

#### 1.2 Concrete Examples

- A ¥10,000 digital game on a platform:  
  - Cannot be resold, gifted to a friend, or included in an estate.  
  - Its value drops to zero when the account disappears.
- Creative software suites:  
  - Users must pay indefinitely or lose access to both tools and proprietary formats.  
  - Long‑term projects are tied to a subscription meter.
- E‑books in proprietary ecosystems:  
  - Account bans have resulted in total loss of purchased libraries.

#### 1.3 Economic Impact

- Physical second‑hand markets in books and games represent **hundreds of billions of yen** annually in Japan alone.  
- Digital ecosystems currently fail to recreate this circulation and reallocation of value.  
- Without resale, the long tail of works is under‑monetized; older works cannot find new audiences through price discovery.

---

### 2. Crisis 2: Capability Externalization via AI and OSS

#### 2.1 From “Money Risk” to “Human‑Rights Risk”

The core issue with modern AI and OSS is no longer primarily financial; it is the ease of exporting dangerous capabilities:

- Non‑consensual deepfake pornography at scale.  
- Automated exploit generation and vulnerability weaponization.  
- Biometric spoofing using cloned faces and voices.  
- Large‑scale political manipulation via convincing synthetic media.

These harms are not abstract—they are already observable. The question is **how to prevent models and tools from becoming persistent, unrevocable weapons once released.**

#### 2.2 Collapse of Classical OSS Assumptions

Classical open‑source thinking assumed:

> “Externalizing harmful capabilities is expensive.  
>  Therefore transparency and peer review are net‑beneficial.”

Today, that assumption is breaking:

- Individual actors can mass‑produce deepfakes in hours.  
- LoRA fine‑tuning can bypass safety layers cheaply.  
- Model composition allows recombining dangerous abilities with little cost.

As a result:

- The **benefits of transparency** (reproducibility, academic scrutiny) no longer automatically outweigh  
  the **risk of capability proliferation** in all domains.

#### 2.3 Limits of Current Mitigations

1. **Responsible AI Licenses (e.g., OpenRAIL‑M)**  
   - List prohibited uses (illegal content, hate speech, etc.).  
   - But once a model is downloaded:
     - Execution cannot be practically prevented.  
     - Violations are hard to detect.  
     - Legal enforcement is slow, costly, and jurisdiction‑bound.

2. **Community Norms and Codes of Conduct**  
   - Rely on goodwill and peer pressure.  
   - Malicious actors ignore norms and simply redistribute on dark markets.

3. **Government Regulation**  
   - Legislative and regulatory cycles lag far behind model development.  
   - International coordination is difficult; enforcement across borders is patchy.  
   - Over‑regulation risks harming benign research and expression.

The conclusion: **license text and social norms alone are insufficient**. We need a technical substrate that can represent and enforce revocable rights at the execution level.

---

### 3. Crisis 3: The Financialization of Blockchain

#### 3.1 Historical Drift

Originally, blockchain research focused on **time‑stamping and public ledgers**:

- 1991 — Haber & Stornetta:  
  - “How to timestamp a digital document”  
  - Use case: tamper‑proof records, public notarization, registry functions.

Later, the focus shifted:

- 2008 — Bitcoin (Nakamoto):  
  - “Peer‑to‑peer electronic cash”  
  - Use case: censorship‑resistant currency.
- 2013–2021 — ICOs, DeFi, NFT waves:  
  - Tokens, leveraged derivatives, speculative art markets.  
  - Emphasis on price action, not public infrastructure.

The original civic potential—**a neutral, verifiable registry layer**—has been overshadowed by speculative use cases.

#### 3.2 Reframing: “Administrative Blockchain”

This proposal re‑centers blockchain as:

- A **public registry** for:
  - Ownership, licenses, and lineage of digital artifacts.  
  - Revocation events (CRLs) and rights assignments.
- An **administrative utility**, analogous to:
  - Land registries.  
  - Corporate registries.  
  - Notary and government printing office functions.

It is not a global casino, nor a fully centralized corporate database, but a semi‑decentralized layer co‑governed by public and private actors.

---

## Part II — Design Principles

### 4. Maker Sovereignty

**Maker sovereignty** is defined as:

- The ability of a creator to:
  - Control how their digital artifacts are licensed and executed.  
  - Revoke or limit dangerous uses.  
  - Participate in economic upside from reuse and resale.
- Without:
  - Permanently locking users into closed ecosystems.  
  - Exercising unilateral, unaccountable power over infrastructure.

This requires:

- _Traceable provenance_ (who created what, when, under which license).  
- _Executable rights_ (keys that actually gate high‑risk code or model execution).  
- _Revocation mechanisms_ (practical capability to disable or downgrade a dangerous artifact).

### 5. Semi‑Decentralization as the Practical Optimum

Fully centralized and fully decentralized architectures both fail in different ways:

- **Pure centralization:**
  - Single points of failure and censorship.  
  - Vendor lock‑in; incentives to maximize rent extraction.
- **Pure decentralization:**
  - No effective revocation or governance.  
  - Difficult to coordinate safety policies; “code is law” rigidity.

The proposal advocates a **semi‑decentralized** model:

- A small number of audited, interoperable registry operators.  
- Transparent rules for registering artifacts and revoking keys.  
- Oversight mechanisms (public reporting, external audit, possibly regulatory interfaces).

In this model:

- The registry is not owned by a single corporate actor.  
- But it is also not an ungoverned, permissionless Wild West.

---

## Part III — Technical Architecture (How)

### 6. High‑Level Flow: Approve → Token → Key → Execute

The core execution flow is:

1. **Approve**  
   - A maker registers a digital artifact (model, dataset, VR asset, library) and an associated license profile.  
   - Safety review and risk classification may be required for high‑risk categories.
2. **Token**  
   - The registry issues signed tokens representing rights and obligations (e.g., right to execute, right to resell).  
   - Tokens are bound to identities and hardware attestation where appropriate.
3. **Key**  
   - Encrypted artifacts (or critical parts) are only usable with keys that correspond to valid tokens.  
   - Keys can be revoked or rotated; revocation events are logged on the registry.
4. **Execute**  
   - Execution environments (TPM/TEE, sandboxed runtimes) verify tokens, keys, and revocation lists before use.  
   - For ordinary low‑risk artifacts, this can be transparent; for high‑risk ones, strict gating is applied.

### 7. Identity, Hardware, and Revocation

- **Identity and attestation**
  - Users and organizations obtain cryptographic identities tied to hardware attestation (TPM/TEE) where needed.  
  - For sensitive capabilities, the system can mandate hardware‑verified execution environments.

- **Revocation lists (CRL)**
  - The registry maintains revocation lists for:
    - Compromised keys.  
    - Artifacts whose risk profile has changed (e.g., new exploits discovered).  
    - Actors banned from certain use cases (e.g., repeat offenders).

- **Separation of concerns**
  - The registry does not host full artifacts; it hosts:
    - Hashes and metadata.  
    - License descriptors.  
    - Revocation and update events.

Actual content distribution remains flexible (CDNs, IPFS, private storage), but must respect keys and policies.

---

## Part IV — Economic Design

### 8. Resale, Royalties, and Cost Sharing

To ensure that safety mechanisms **do not** undermine legitimate economic activity, the system includes:

- **Resale and gifting support**
  - Tokens can be transferred, sold, or gifted, subject to license constraints.  
  - Makers can specify resale royalty rates, ensuring ongoing revenue without subscription lock‑in.

- **Sustainability via cost sharing**
  - Registry and safety operations (vetting, audits, CRL maintenance) incur costs.  
  - A fraction of transaction fees and royalties funds these operations, aligning safety with economic incentives.

- **Aligning incentives**
  - Makers earn more when artifacts are widely reused **within safe constraints**.  
  - Users benefit from genuine ownership (ability to resell/gift) rather than perpetual rent.

In this design, **safety is not an external tax** imposed on the system; it is a byproduct of how value flows:

- Dangerous capabilities require higher assurance and clearer provenance, which justifies slightly higher fees
  and stricter gating.  
- Benign capabilities flow with minimal friction, but still benefit from provenance and resale support.  
- Makers and platforms gain financially when they keep artifacts inside the “safe reuse” region; pushing
  artifacts into obviously abusive territory risks revocation and loss of revenue.

### 9. Compatibility with Existing Ecosystems

- The model is intended to wrap, not replace, existing code hosting and package ecosystems (git, npm, PyPI, model hubs).  
- For high‑risk capabilities, those ecosystems can:
  - Require registry tokens for download or execution.  
  - Integrate revocation signals to de‑list or disable known‑dangerous artifacts.

---

## Part V — Application Scenarios (What)

### 10. VR and Games: Ownership‑Centric

Goals:

- Restore the ability to **buy, gift, and resell** virtual goods.  
- Prevent “total loss” scenarios when a platform fails or an account is banned.

Illustrative behavior:

- A user purchases a VR item as a tokenized right; the item can later be:
  - Resold in a secondary market.  
  - Gifted to another user.  
  - Bequeathed as part of an estate.
- Makers receive a modest resale royalty, ensuring ongoing income without subscriptions.

### 11. Commercial AI: Safety‑Centric

Goals:

- Control the externalization of dangerous capabilities (e.g., exploit generation, biometric spoofing).  
- Allow revocation and downgrading of models that cross safety thresholds.

Illustrative behavior:

- A model with high abuse potential is distributed only as an encrypted artifact.  
- Execution requires:
  - Valid tokens verifying the use case and actor.  
  - Hardware‑backed attestation for critical deployments.  
  - Ongoing CRL checks to ensure the key has not been revoked.

When new abuse patterns are discovered:

- The registry can flag the artifact, revoke or narrow its keys, and require updated safety filters.

### 12. OSS Research: Safety and Openness

Goals:

- Preserve the benefits of open research (reproducibility, peer review)  
  while reducing the risk of uncontrolled capability proliferation.

Approach:

- **Parallel distribution:**  
  - Source code and low‑risk components may remain openly accessible.  
  - High‑risk modules (models, exploit generators) are gated by the registry.

- **Dual‑key model:**  
  - Researchers may obtain keys for experimentation under controlled conditions.  
  - Deployment or commercialization keys require stricter review and compliance.

This enables a gradient between fully open and tightly controlled, instead of an all‑or‑nothing choice.

---

## Part VI — Limitations and Open Questions

The proposal is intentionally ambitious and faces several challenges:

- **Interoperability and standardization**
  - Defining interoperable token and license formats across ecosystems.  
  - Coordinating multiple registry operators across jurisdictions.

- **Governance**
  - Who operates the registry infrastructure?  
  - How are policies decided and updated?  
  - How can abuse by powerful actors be prevented?

- **Privacy**
  - Balancing provenance transparency with user privacy rights.  
  - Avoiding surveillance while enabling accountability.

- **Adoption and incentives**
  - Encouraging platforms and vendors to adopt the model voluntarily.  
  - Providing clear economic benefits relative to status quo approaches.

These questions are not fully resolved, but the proposal provides a technical and economic sketch within which they can be negotiated.

---

## Part VII — Open Strategy and Patent Pledge

The proposal envisions:

- **Open standards** for:
  - Metadata and token formats.  
  - License descriptors and safety classifications.  
  - API schemes for revocation and provenance queries.

- **Reference implementations**
  - Open‑source prototypes of registry nodes, clients, and integration libraries.  
  - Example adapters for VR platforms, model hubs, and package managers.

- **Patent pledge (if applicable)**
  - To the extent any ideas are patentable, a pledge is made that they will not be used to block interoperable implementations that respect core safety and ownership principles.

The ultimate goal is not a proprietary product, but a **public infrastructure blueprint** that:

- Respects makers’ sovereignty and users’ rights.  
- Tames the worst risks of AI and OSS capability externalization.  
- Reclaims blockchain and cryptographic infrastructure as instruments of civic order, not speculation.
