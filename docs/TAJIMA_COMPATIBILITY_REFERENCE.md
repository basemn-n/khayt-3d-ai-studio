# Tajima Compatibility Reference — Khayt 3D

**Reference revision:** 2026-11-25  
**Status:** Initial verified baseline; not a complete catalogue of every historical, regional, or discontinued Tajima machine.

## Purpose and safety rule

Khayt 3D prepares embroidery geometry and stitch records. The workshop/operator remains responsible for selecting the physically installed frame, mounting the garment, checking machine-specific limits, and setting mechanical conditions.

This reference deliberately separates:

1. **Machine embroidery field** — the manufacturer's published maximum field for a particular model/configuration.
2. **Frame embroidery area** — the area published for a named frame; frame compatibility is model-dependent.
3. **Artwork dimensions** — the requested design width and height in millimetres.
4. **Safe usable area** — must not be inferred from the nominal field. It depends on the installed frame, presser foot, product, clamps, and operating conditions.

Tajima explicitly states that usable field and speed may vary by model, product, and operating conditions, and that actual stitchable area may be limited by the frame and presser foot. Do not treat the maximum field as an unconditional design canvas.

## Verified model records

All dimensions below preserve the manufacturer's stated order and labels. Where the source says D×W, D means depth and W means width. Do not silently swap axes.

| Model / family | Verified manufacturer data | Source |
|---|---|---|
| SAI | 8 needles; maximum field D 200 × W 300 mm. Large tubular frame 200 × 300 mm; small tubular frame 100 × 100 mm (2025 brochure). | [SAI product page](https://www.tajima.com/product/sai/), [SAI 2025 brochure](https://www.tajima.com/wp-content/uploads/2025/06/SAI_EN_PDF_2506.pdf) |
| TMBP2-SC | 6/12/15 needles; field 360 × 500 mm; wide cap 75 × 360 mm; semi-wide cap 83 × 180 mm; tubular 360 × 500 mm; pocket frame 45 × 80 mm (published table). | [TMBP2-SC](https://www.tajima.com/product/tmbp2-sc/) |
| TMBR2-SC | 6/12/15/18 needles; field 360 × 500 mm; tubular 360 × 500 mm; cap 75 × 360 mm; pocket 45 × 80 mm and 80 × 55 mm; further optional frames listed in official brochure. | [TMBR2-SC](https://www.tajima.com/product/tmbr2-sc/), [TMBR2/TMBP2 brochure](https://www.tajima.com/wp-content/uploads/2024/04/TMBR2BP2_E.pdf) |
| TMBP2-XC | 12/15 needles; field 550 × 600 mm; wide cap 75 × 360 mm; semi-wide cap 83 × 180 mm; tubular 550 × 600 mm; pocket 65 × 100 mm or 100 × 75 mm; optional X-extension 550 × 1,200 mm. | [TMBP2-XC](https://www.tajima.com/product/tmbp2-xc/) |
| TMEZ-SC | 9/12/15 needles; field 360 × 500 mm; cap 75 × 360 mm; semi-wide cap 83 × 180 mm; tubular 335 × 453 mm; pocket 45 × 80 mm or 80 × 55 mm. | [TMEZ-SC](https://www.tajima.com/product/tmez-sc/) |
| TMEZ-KC | 9/12/15 needles; 2/4/6/8 heads; head interval 360 or 500 mm. Configurations include field 450 × 360 mm or 450 × 500 mm; cap 75 × 360 mm; semi-wide cap 83 × 180 mm; tubular 439 × 279 mm or 439 × 419 mm. | [TMEZ-KC](https://www.tajima.com/product/tmez-kc/), [official TMEZ-KC specification PDF](https://www.tajima.com/wp-content/uploads/2024/04/TMEZ-KC.pdf) |
| TMER-KC | 9/12/15 needles; 2/4/6/8 heads; 360/500 mm head interval; published configurations include field 450 × 360 mm or 450 × 500 mm, cap 75 × 360 mm, semi-wide cap 83 × 180 mm, tubular 439 × 279 mm or 439 × 419 mm. | [TMER-KC](https://www.tajima.com/product/tmer-kc/) |
| TMAR-KC | 9/12/15 needles; 2–8 heads; published configurations include 450 × 360 mm or 450 × 500 mm field and tubular 439 × 279 mm or 439 × 419 mm; verify exact head/interval configuration. | [TMAR-KC](https://www.tajima.com/product/tmar-kc/) |
| TMAR-VC | 12-head series; published configuration shown with 450 × 360 mm field and tubular 439 × 279 mm; confirm exact variant before use. | [TMAR-VC](https://www.tajima.com/product/tmar-vc/) |
| TFMX-II C | Multi-head cylinder family; published configurations vary by head count and head interval. Official Type-2 PDF lists border 450 × 360/500 mm, tubular 439 × 279/419 mm, cap 75 × 360 mm, semi-wide cap 83 × 180 mm, pocket 65 × 100 or 100 × 75 mm where supported. | [TFMX-II C](https://www.tajima.com/product/tfmx-2c/), [official Type-2 specification PDF](https://www.tajima.com/wp-content/uploads/2024/04/TFMX-2CTYPE2_E.pdf) |
| TMCR-VF / TMCR-VF+i-TM | Flat multi-head families. Field is configuration-specific; official tables list model-dependent dimensions, head intervals and continuous-pattern overlap. Do not assign a single universal field to the family. | [TMCR-VF](https://www.tajima.com/product/tmcr-vf/), [TMCR-VF+i-TM](https://www.tajima.com/product/tmcr-vf-itm/) |
| TMCP-VF | Flat multi-head family; field and head interval are configuration-specific. Read the exact model row from the official product table. | [TMCP-VF](https://www.tajima.com/product/tmcp-vf/) |
| TMCS-VF | Flat multi-head family listed in Tajima's current product catalogue; detailed model-specific field data still needs to be captured from its current official catalogue/manual. | [Tajima machine catalogue](https://www.tajima.com/product/) |

## Frame compatibility is not universal

Tajima's frame catalogue explicitly says frame compatibility differs by machine model and directs users to confirm sizes and compatible machines. The listed frame families include border, tubular, M, cap, pocket, sock, rotary-knob clamp, air-type clamp, air-type pocket, and X-extension. Therefore Khayt 3D must not display every frame as compatible with every machine.

Source: [Tajima embroidery frames](https://www.tajima.com/product/frame/).

## Digitizing settings: what Khayt 3D may and may not assume

Tajima DG16 documentation describes fabric recipes and per-segment controls such as underlay, density, push/pull compensation, stitch types, sequence editing, and stitch settings. It does **not** justify one universal density, pull compensation, underlay, stitch length, or tension value for every fabric and design.

Sources: [Tajima DG16 product/software information](https://fonts.pulsemicro.com/Home/Embroidery), [DG16 catalogue PDF](https://tajima.com.vn/wp-content/uploads/2021/03/TAJIMA-DG16.pdf).

Implementation rules:

- Keep fill density, satin density, underlay, pull compensation, stitch length, and tie-in/tie-off as explicit per-object/per-recipe properties.
- A recipe must identify fabric class, stitch type, source document/version, and whether its values are manufacturer-published, software-preset-derived, operator-entered, or test-calibrated.
- Until exact numeric values are verified from an authoritative document or a controlled sew-out, store them as `null` / `not_verified`; never silently substitute guessed defaults.
- Do not infer upper-thread tension, bobbin tension, presser-foot pressure, needle size, thread type, or machine speed from a DST file. These depend on the machine, material, consumables, and operator setup.
- Do not call a design “production-ready” solely because its geometry passes software checks. Use “software checks passed — physical sew-out required” until an actual sample has been approved.

## DST handoff contract

DST is the stitch-data handoff file, not a complete machine setup package. Khayt 3D should provide a companion production sheet containing at least:

- design width and height (mm), orientation, and origin/center reference;
- stitch count, color-change/stop count, jump count, trim count if reliably available;
- stitch-type summary (running, tatami/fill, satin, etc.);
- sequence and color chart (brand/thread color mapping is a separate human-readable aid);
- detected long stitches, short stitches, dense overlaps, and other QA warnings;
- fabric/recipe identifier and its evidence status;
- a statement that the operator must verify the installed compatible frame and run a sample.

Do not claim that DST preserves thread brand names, exact color names, fabric recipe, hoop selection, or all machine-specific controller settings.



## Priority workflow: full garment panels, borders, and sleeves

Khayt 3D's primary production target is embroidery on **abaya/coat panels and large fabric pieces**, especially full-panel work. Secondary targets are border/hem embroidery and sleeve/hand sections. This affects machine-family guidance:

- **Cut fabric before garment assembly / full panel / large continuous artwork:** prioritize Tajima **flat-type** machines and border frames. Tajima's machine manual classifies flat-type machines as suitable for embroidery on cut or original fabric. The official frame catalogue describes border frames as intended for fabric before it is made into products and for large-lot items such as curtains, bedding, pillow cases, and patches.
- **Finished garment, tubular access, sleeves or constrained garment areas:** cylinder-type machines may be appropriate where the garment can be mounted on the compatible tubular/frame system. Actual access and field depend on garment construction and model.
- **Border / hem:** model- and frame-specific border-frame workflow; long borders may require repeated positioning or a continuous-pattern function. Do not automatically split a long border into segments until overlap/registration rules for the exact machine and frame are known.
- **Sleeves / hand sections:** treat as a separate placement job. The design file should retain an origin/center reference and mirrored variant where needed; frame compatibility and clearance must be checked for the exact machine.
- **Full panel coverage:** artwork dimensions must be captured in mm from the actual cut piece or approved pattern. The machine's nominal field is not the garment size. If the artwork exceeds the available field, Khayt 3D must flag it and offer manual panel segmentation only after the operator confirms the frame, repositioning method, and registration marks.


### Full-energy fabric-first cutting layout (confirmed production workflow)

The primary workflow is **embroider the complete flat fabric piece first, then cut the garment pieces from that embroidered fabric**. The incoming fabric is treated as one shared workpiece, not as separate already-cut sleeves/front/back parts.

The layout planner must support a coordinated cut plan on the same fabric sheet, with independently placeable garment-piece regions such as:

- the long border/hem band;
- left and right sleeve/hand pieces (paired and mirrored where the pattern requires);
- back panel;
- left and right front panels;
- any additional pieces explicitly included by the tailor/cutter.

The operator or cutter supplies/approves the actual garment pattern, piece dimensions, grain direction, seam allowances, and placement. Khayt 3D must not invent garment measurements or assume a universal abaya pattern.

**Required separation of geometry layers:**

1. **Stitch artwork:** the actual embroidery paths and stitch objects.
2. **Cutting outlines:** non-stitch guides showing where the cutter will cut each piece after embroidery.
3. **Placement/registration guides:** non-stitch marks or references used to align the fabric and repeat/segment work.
4. **No-stitch / seam allowance zones:** protected areas where embroidery must be kept clear unless the user explicitly overrides them.

Cutting outlines and registration guides must never be exported as embroidery stitches. They must remain visible in the layout preview and in a separate cutting/layout report. The stitch export contains only embroidery objects.

The layout planner must:
- place all garment pieces on the same fabric sheet and show the remaining usable fabric;
- allow the border to be embroidered as a continuous or repeated strip, with explicit start/end margins and approved joins;
- keep embroidery inside the approved region of each cut piece and away from cut lines/seam allowances by the user-specified clearance;
- prevent unintended stitch artwork from crossing between two pieces that will later be cut apart;
- preserve grain direction and orientation, and flag mirrored left/right pieces for explicit confirmation;
- show piece labels, dimensions, rotation, placement coordinates, and the overall fabric-sheet dimensions;
- detect overlap between cut pieces, cutting paths, and stitch artwork, and warn before export;
- support a manual cutter-approved layout rather than automatically claiming an optimal fabric nest.

**Production sequence:** (1) enter actual fabric-sheet dimensions; (2) import or draw/approve the garment cutting pattern; (3) arrange the border, sleeves, back, and front pieces; (4) place embroidery within each piece/strip; (5) review cut lines, seam allowances, joins, and registration; (6) export the stitch file plus a separate cutting/layout sheet; (7) embroider the fabric; (8) cut the pieces and sew the garment.

The workshop still chooses the physical frame and machine setup. If the whole sheet or a region exceeds the verified field of the actual machine/frame, the software must flag it and require an operator-approved repositioning/registration plan; it must not silently split, scale, or crop the artwork.

### Production presets in the UI

Add named workflow choices, not invented stitch recipes:

1. `FULL_PANEL` — full cut fabric / full panel; requires panel width × height, placement origin, fabric class, and operator-confirmed frame.
2. `BORDER_HEM` — border or hem; requires border length × width, path direction, start/end margins, and whether the machine workflow supports continuous/repositioned sewing.
3. `SLEEVE_HAND` — sleeve/hand section; requires left/right selection, placement dimensions, and mirror confirmation.
4. `GARMENT_PANEL` — front/back/specific garment panel; requires panel identifier and dimensions.

These are **design workflow profiles**, not claims that a particular Tajima machine supports a given garment or frame. The UI must show machine family (flat/cylinder), exact model/configuration, frame compatibility status, and evidence source separately.

### Tajima source basis for this priority

- Tajima machine operation manual: flat-type machines are suitable for cut/original fabric; cylinder-type machines are suitable for cylindrical products such as completed clothes and caps. [Official Tajima machine manual (PDF)](https://www.tajima.support/Manual_machine/%E8%8B%B1%E8%AA%9E/M-EMB-05-EN.pdf)
- Tajima frame catalogue: border frame is a standard frame for flat-type machines and is intended for fabric before product assembly; compatibility varies by machine model. [Official frame catalogue](https://www.tajima.com/product/frame/)
- Tajima TMCP-VF: flat-type machine intended for fabrics and parts before sewing, including large-lot apparel. [Official TMCP-VF page](https://www.tajima.com/product/tmcp-vf/)
- Tajima TMCR-VF: flat-type machine for stable mass production; DCP helps hold material across thickness changes. [Official TMCR-VF page](https://www.tajima.com/product/tmcr-vf/)
- Tajima TMCS-VF: official page lists concrete model configurations and embroidery fields; do not generalize one row to all models. [Official TMCS-VF page](https://www.tajima.com/product/tmcs-vf/)

### Required full-panel QA additions

- Require panel width/height in mm and distinguish fabric-piece dimensions from intended stitch area.
- Show artwork bounding box, margins, center/origin, and orientation.
- Warn when artwork approaches or exceeds the selected model/frame field; never silently scale or crop.
- For repeated or segmented panels, require operator-approved registration/overlap strategy.
- For paired sleeves, verify left/right mirroring and placement separately.
- Estimate stitch count and production complexity, but do not infer runtime from stitch count alone.
- Continue to require a physical sample; full-panel work can reveal fabric movement, distortion, hooping and registration issues not caught by geometry checks.

## Coverage status and next research work

This first pass captures the currently listed Tajima product families and records a subset of model-specific specifications verified from official product pages or official brochures. It is **not yet an exhaustive compatibility database** for every legacy/discontinued machine, regional option, serial-number revision, or accessory.

Before enabling automatic frame recommendations in the UI, complete these tasks:

1. Extract every configuration row from official PDFs for all current product families.
2. Capture model codes, needle count, head count, head interval, field D×W, frame types/sizes, and optional extension limits as separate fields.
3. Add discontinued/legacy models only when official documentation is available.
4. Store source URL, document title, publication/revision date, access date, and page/table reference for every numeric field.
5. Add a compatibility matrix linking each frame SKU to the exact supported machine model/configuration.
6. Add a validation state: `verified`, `partial`, `unknown`, or `operator_confirmation_required`.

## Official sources

- [Tajima current embroidery machine product catalogue](https://www.tajima.com/product/)
- [Tajima frame catalogue and compatibility notice](https://www.tajima.com/product/frame/)
- [Tajima DG software documentation/downloads](https://cloud.pulsemicro.com/Download/Documentation)
- [Tajima DG16 software feature reference](https://fonts.pulsemicro.com/Home/Embroidery)
