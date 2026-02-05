### Phase 1 — **Dataset & Problem Validation (MANDATORY)**

⏱️ ~30–60 minutes
❗ Skipping this is the #1 reason GAN projects fail

#### What you do now

Before writing **any model code**, you must **prove**:

1. **Pairs are truly aligned**

   * Same crop
   * Same resolution
   * Same orientation
   * No pixel shift

2. **Data consistency**

   * Image size distribution
   * Color channels (RGB vs grayscale)
   * File format consistency
   * Dynamic range (0–255 vs 0–1)

3. **Train / validation split**

   * No leakage (same location in both sets)
   * Typical: 80 / 20

#### Output of Phase 1

* A short **data sanity notebook or script**
* Sample visualizations:

  ```
  [Satellite | Map]
  ```

👉 **Only after this phase passes, you move on.**

---

### Phase 2 — **System & Model Design (NO CODE YET)**

⏱️ ~45 minutes

You design **on paper** (or markdown):

#### Decisions you must lock

1. **Image resolution**

   * 256×256 or 512×512 (choose based on GPU)
2. **Normalization strategy**

   * [-1, 1] (pix2pix standard)
3. **Generator**

   * U-Net depth
   * Number of filters
4. **Discriminator**

   * Patch size (e.g. 70×70 PatchGAN)
5. **Loss weights**

   * λ_L1 (typically 100)
6. **Metrics**

   * PSNR
   * SSIM

#### Output of Phase 2

* A **DESIGN.md** or section in README
* Clear architecture diagram (even ASCII is fine)

👉 This prevents **re-architecting mid-training**, which is expensive.

---

### Phase 3 — **Project Skeleton & Infrastructure**

⏱️ ~30 minutes

Now you create **structure, not models**.

#### What you implement

* Folder structure
* Config file (YAML / Python dict)
* Dataset loader (no training yet)
* Logging utilities
* Checkpoint system

At this stage:
❌ No GAN logic
❌ No training loop

You are building **infrastructure**, like a real ML team.

---

### Phase 4 — **Minimal Baseline (VERY IMPORTANT)**

⏱️ ~1–2 hours

Before GANs, you do this:

#### Baseline model

* Generator only
* L1 or L2 loss
* No discriminator

Why?

* Verifies:

  * Data loading
  * Loss correctness
  * Metrics computation
* Gives you a **baseline PSNR / SSIM**

This step is **pure gold** in reports and interviews.

---

### Phase 5 — **Full pix2pix GAN Implementation**

⏱️ Main development phase

Now you:

* Add discriminator
* Add adversarial loss
* Combine losses
* Train carefully

You already know everything works because of Phases 1–4.

---

### Phase 6 — **Evaluation, Visualization & Reporting**

* Metric tables
* Failure cases
* Visual results
* Comparison to baseline

### Phase 7 - **Model Refinement**
This is a **very good** question—and honestly, this is the exact moment where people move from *“I can run a model”* to *“I understand modeling”*. So you’re not missing something trivial; you’re hitting the **core difficulty of advanced CV**.

I’ll answer this in a **principled, systematic way**, not with random tips.

---

#### The core mindset shift (this is the key)

👉 **Model refinement is NOT “try random hyperparameters”**
👉 **Model refinement is hypothesis-driven debugging**

Think of your pix2pix cGAN as a **system** that can fail in *specific ways*.
Your job is to:

> **Observe failures → classify them → apply targeted changes**

If you don’t know *what kind* of failure you’re seeing, changing things is blind.

---

#### Step 1: Define *what “better” means* (before touching anything)

For image translation, “accuracy” is vague. You must decide **which axis matters**:

For pix2pix (satellite → map), typical objectives are:

* Structural alignment (roads in the right place)
* Sharp edges (no blurry maps)
* Semantic correctness (water ≠ roads)
* Visual realism

And metrics you already know:

* **L1 / L2**
* **SSIM**
* **PSNR**
* * qualitative inspection (very important for GANs)

📌 **Rule**:
If you don’t know *what kind of improvement you want*, you cannot refine.

---

#### Step 2: Diagnose the failure mode (this is the most important step)

When you look at generated images, ask **specific diagnostic questions**:

##### A. Are outputs **blurry but structurally correct**?

➡️ Typical cause:

* Generator relies too much on L1 loss
* Adversarial signal too weak

🎯 Candidate actions:

* Increase GAN loss weight (λ_GAN ↓ λ_L1)
* Stronger discriminator (PatchGAN size ↑)
* Add perceptual / SSIM loss

---

##### B. Are outputs **sharp but wrong / unstable**?

(e.g., hallucinated roads, artifacts)

➡️ Typical cause:

* Discriminator overpowering generator
* Training instability

🎯 Candidate actions:

* Lower discriminator LR
* Use label smoothing
* Add spectral norm to D
* Reduce PatchGAN aggressiveness

---

##### C. Are outputs **structurally wrong**?

(roads shifted, missing regions)

➡️ Typical cause:

* Receptive field too small
* Generator capacity insufficient

🎯 Candidate actions:

* Deeper U-Net
* Larger PatchGAN (70×70 → 142×142)
* Less aggressive cropping / augmentation

---

##### D. Does training **collapse or oscillate**?

(loss spikes, mode collapse)

➡️ Typical cause:

* Bad training dynamics

🎯 Candidate actions:

* TTUR (different LR for G and D)
* Gradient penalty
* Fewer D steps per G step

---

#### Step 3: Refine in the **correct order** (VERY IMPORTANT)

##### ❌ Wrong order:

> Change architecture + optimizer + loss + data all at once

##### ✅ Correct order (almost always):

##### 1️⃣ **Data sanity first**

Before touching the model:

* Input/output perfectly aligned?
* Normalization correct? ([-1,1] for pix2pix)
* Augmentation realistic? (NO rotations for maps unless valid)

📌 Many “bad models” are actually **bad data**.

---

##### 2️⃣ **Training dynamics**

Only once data is solid:

* Learning rates (G vs D)
* Batch size
* GAN vs L1 balance
* Stability tricks

This fixes **most GAN problems**.

---

##### 3️⃣ **Loss function**

Ask:

* Do I want structure? → L1 / SSIM
* Do I want realism? → GAN weight ↑
* Do I want semantics? → perceptual loss

Change **ONE loss term at a time**.

---

##### 4️⃣ **Architecture (last, not first)**

Only change architecture if:

* Model clearly underfits
* Structural errors persist

Architecture is **expensive and slow to debug**.

---

#### Step 4: One change = one hypothesis

Every refinement should answer this sentence:

> **“I believe the model fails because X, so I will change Y to test it.”**

Examples:

* “Images are blurry → adversarial signal too weak → increase GAN loss”
* “Edges missing → PatchGAN too local → increase patch size”
* “Roads inconsistent → generator lacks context → deeper U-Net”

If you cannot say this sentence, **don’t change anything**.

---

#### Step 5: Keep a refinement log (this is pro-level)

Literally write:

```
Experiment 07
Change: λ_L1 from 100 → 50
Hypothesis: Reduce blur, increase sharpness
Result: Sharper edges, slight artifacts
Decision: Keep λ_L1=75
```

This is how **researchers** work.
Without this, refinement becomes chaos.

---

#### Step 6: For pix2pix specifically (your project)

Given your setup (64×64 → 600×600 tiles, PatchGAN, SSIM, etc.), your **highest-leverage refinements** are:

1. **λ balance (GAN vs L1 vs SSIM)**
2. **PatchGAN size**
3. **Data augmentation realism**
4. **Discriminator strength vs Generator**
5. **Evaluation on held-out regions (not random tiles)**

Not random architecture changes.

---

##### The mental model to remember

Think of refinement as:

> **Debugging a learning system, not tuning knobs**

You don’t “optimize everything”
You **identify bottlenecks and remove them one by one**.



### phase 8 - **Extra Work** 

* UI (handcrafted)

  - Flask (Python backend)

  - HTML/CSS/JavaScript frontend

  - File upload form

* UI (raedy made tool)
   - Gradio