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


### phase 7 - **Extra Work** 

* UI (handcrafted)

  - Flask (Python backend)

  - HTML/CSS/JavaScript frontend

  - File upload form

* UI (raedy made tool)
   - Gradio