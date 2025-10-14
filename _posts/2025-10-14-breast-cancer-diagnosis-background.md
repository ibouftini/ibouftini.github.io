---
title: "Background on Breast Cancer Diagnosis"
date: 2025-10-14
permalink: /posts/2025/10/breast-cancer-diagnosis-background/
tags:
  - breast cancer
  - medical imaging
  - mammography
  - computer-aided diagnosis
  - healthcare
---

In this section, we will introduce a brief description of the core challenges in Breast Cancer detection.

## Medical context

Breast cancer is one of the most common oncological diseases in the world with **2.3 million new cases identified each year**. It arises either from hereditary genetic factors or lifestyle practices. Anatomically, breast cancer is caused by the uncontrolled multiplication of breast cells. Depending on the biological behavior of these cells, there are two forms of tumor: benign and malignant (Figure 1). Benign tumors are formed by cells that partially retain their morphology and function, although they continue to multiply. Malignant tumors, also called cancers, on the other hand, are characterized by a morphology and function that are different from those associated with healthy tissues. During their reproduction, they form extensions that infiltrate adjacent tissues, enveloping normal cells and destroying them, a phenomenon known as neoplastic invasiveness.

![Examples of benign and malignant masses](/images/breast-cancer-diagnosis/mass.png)
*Figure 1: Examples of benign and malignant masses. Reproduced from Alghaib et al.*

Early diagnosis on screening programs is the most effective tool for reducing mortality associated with neoplasms. Cancer screening involves conducting surveillance tests on a person who is supposedly healthy, with the aim of detecting abnormalities that could be warning signs of cancer, well before the first symptoms appear. This approach has proven highly effective in that it significantly reduces the mortality rate from breast cancer by improving the chances of recovery. Indeed, if detected early, breast cancer can be cured in nine out of ten cases, contributing to a 15 to 21% reduction in the mortality rate from this cancer.

## Mammography

### Presentation

The gold standard in screening programs is represented by mammography, that is, a precise and reliable diagnostic test, capable of detecting nodular lesions, even small ones, not yet detectable by touch. Mammography is also able to detect the presence of microcalcifications (small calcium deposits due to the secretions of mutated cells) that can be an indication of precancerous lesions. Distribution patterns of calcifications can be:

- **Diffuse**: Random distribution throughout breast tissue → benign
- **Regional**: significant tissue proportion (>2cm) → likely benign
- **Cluster**: Five or more calcifications in 1-2cm area → malignancy risk
- **Linear**: Arranged in ductal pattern → potential malignancy
- **Segmental**: Deposits in duct systems and branches → malignancy

![Distribution of calcifications](/images/breast-cancer-diagnosis/calcifications.png)
*Figure 2: Distribution of calcifications. Reproduced from Mračko et al.*

Mammography employs an X-ray beam using an X-ray tube, oriented so as to be tangent to the patient's sternum (see Fig. 3). X-rays are radiations capable of penetrating inside biological tissues and reaching the detector. The quantity of radiation absorbed by the body depends on the type of object being examined and in particular it is greater the denser the body crossed is. Mammographs use two compression plates to distribute breast tissue so as to optimize image quality.

### X-ray beamforming

When high-speed electrons strike a metal target, their kinetic energy is transformed into heat (99%), and into X-rays (1%). An X-ray tube consists of three elements: an electron source (cathode), a potential difference, and a metal target for generating X-rays (anode). The assembly is contained within a vacuum chamber and a leaded sheath containing a window that allows the X-ray beam to pass through.

![Mammograph design](/images/breast-cancer-diagnosis/mammo.png) ![X-ray tube design](/images/breast-cancer-diagnosis/mammo2.png)
*Figure 3: (a) Illustration of X-ray tube design (b) Illustration of mammograph design. Adapted from Radiology Cafe*

The cathode consists of one or two filaments to create an electron source and a concentrating (or focusing) part that accommodates and holds the filament(s) in place. The electron source is obtained by the thermionic effect where the tungsten filament is heated to incandescence and the heat is transmitted to the free electrons in the metal in the form of kinetic energy. Thanks to this energy gain, the electrons are torn from the filament and form an electron cloud around the filament called a space charge. The electrons located around the filament are attracted to the target by a high potential difference (40 to 150 kV).

The purpose of the vacuum is to allow precise and separate control of the number and speed of accelerated electrons. If gas were present inside the tube, the electrons accelerated toward the anode would collide with the gas molecules, causing them to lose kinetic energy and triggering the formation of secondary electrons ejected from the gas molecules by ionization. This would cause large variations in the tube current intensity and the energy of the X-rays produced.

Regarding the anode, it must be sufficiently dense to promote the production of X-ray (braking effect), have a high melting temperature to withstand the temperatures secondary to electronic interactions, and be a good thermal conductor to quickly dissipate heat.

The **Beer-Lambert Law**, adjusted for the varied composition of breast tissue, provides the basic formula determining X-ray attenuation in mammography:

$$I(d) = I_0 \cdot \exp\left(-\sum_{i} \mu_i(E) \cdot d_i\right)$$

where:
- $I(d)$ = Transmitted X-ray intensity after traversing breast thickness $d$ (photons/mm²)
- $I_0$ = Incident X-ray intensity (photons/mm²)
- $\mu_i(E)$ = Energy-dependent linear attenuation coefficient for tissue type $i$ (cm⁻¹)
- $d_i$ = Thickness of tissue type $i$ (cm)

For a breast composed of adipose ($a$) and glandular ($g$) tissues:

$$I = I_0 \cdot \exp\left(-[\mu_a(E) \cdot d_a + \mu_g(E) \cdot d_g]\right)$$

The attenuation equation explains why X-rays lose intensity as they pass through breast tissue. This concept is used in practice by radiologists to adjust imaging parameters. For example, longer exposure times or higher beam intensities are required in the case of highly glandular breast tissue because it absorbs more radiation.

Four standardized projections make up the mammography examination protocol: mediolateral oblique (MLO) views, which provide oblique viewpoints, and craniocaudal (CC) views, which capture top-to-bottom breast imaging. These views can be described for each breast.

The MLO view highlights a greater region of breast tissue including the superior and lateral regions of the breast as well as the region next to the armpit. It is typically the first view taken during a diagnostic or screening mammography. By observing the pectoral muscle, this technique makes it easier to evaluate the breast position and image clarity. Viewing the breast tissue from superior to inferior is possible with the CC view. It works best for examining the inside of the breast and enhances the image produced by the MLO view. It provides the chance to examine the breasts features and verify that there is no distortion.

![Relations between CC and MLO views](/images/breast-cancer-diagnosis/ccmlo.png)
*Figure 4: Relations between CC and MLO views. Reproduced from Liu et al.*

## Other imaging tools

Tomosynthesis, also called Digital Breast Tomosynthesis (DBT), is a sophisticated mammography method in which the X-ray tube rotates in an arc across the breast to take several low-dose pictures from different angles usually ±25° (see Fig. 5). Compared to conventional 2D mammography, studies show higher cancer detection rates in DBT thanks to its 3D layered images which reduce tissue overlap.

![Illustration of Tomosynthesis](/images/breast-cancer-diagnosis/tomo.png)
*Figure 5: Illustration of Tomosynthesis. Reproduced from Kontos et al.*

Magnetic Resonance Imaging uses radio waves and magnetic fields to create a detailed picture of breast tissue. It uses a large, tube-shaped equipment that produces intense magnetic fields for detection (see Fig. 6). The main drawback is the rise in false-positive rates, which may call for needless follow-up testing and raise patient worry and medical expenses. Hence, MRI is often employed as an additional screening technique for high-risk patients.

![MRI imaging systems and results](/images/breast-cancer-diagnosis/mri.png) ![Breast MRI scan](/images/breast-cancer-diagnosis/mri2.png)
*Figure 6: (a) Illustration of MRI tube design (b) Breast MRI scan. Sources: (a) reproduced from Radiology Cafe, (b) reproduced from Joines et al.*

Despite the enhanced diagnostic capabilities of MRI and tomosynthesis, scalable deployment across healthcare systems is restricted by **high costs and restricted accessibility**. These technologies restrict throughput in high-volume screening settings since they call for specific tools, skilled workers, and lengthy examination periods. Mammography is therefore a commonly used method for diagnosing breast cancer because it balances diagnostic accuracy with operational feasibility.

## Need for CAD systems

Mammographic interpretation presents significant challenges in clinical practice that have motivated the development of Computer-Aided Detection (CAD) systems. These challenges stem from both the inherent complexity of mammographic images and systematic healthcare delivery issues.

### Workload of Radiologists and Resource Limitations

There is a severe lack of skilled radiologists in the world's healthcare system which highly influences the precision of diagnosis. The Association of American Medical Colleges projects a shortage of 17,000 to 42,000 radiologists, pathologists, and psychiatrists by 2033. Current indicators of this shortage include over 1,400 open radiologist positions on the American College of Radiology's job board and reports that 53% of practicing radiologists are age 55 or older. Research indicates that radiologists who work above their usual capacity encounter significantly higher rates of diagnostic mistakes; workload normalized scores on days when diagnostic errors occurred average 121% higher than baseline levels.

Radiologist workload has skyrocketed in recent years; according to one European research, on-call workload doubled over a 15-year period in terms of relative value units. In screening programs, when a high volume of primarily normal patients must be handled quickly, this workload strain is especially severe.

### Breast Density Problem

Macroscopically each breast has three main components: the mammary gland, the skin and the nipple-areola complex (see Fig. 7). The gland is surrounded by a dense network of connective tissue including arteries, veins, nerves and lymphatic vessels.

![Breast anatomy illustration](/images/breast-cancer-diagnosis/breast.png)
*Figure 7: Breast anatomy illustration. Reproduced from Johns Hopkins Pathology*

Similar to cancers, dense breast tissue appears white on mammograms. Because of this, it becomes difficult to distinguish between normal and cancerous tissue in dense breasts using mammography.

ACR BI-RADS Atlas 2013 standardization framework utilizes categories a-d based on fibroglandular tissue, also called dense tissue, likelihood to obscure masses:

- **Category 1:** Almost entirely fatty breasts
- **Category 2:** Scattered fibroglandular density areas
- **Category 3:** Heterogeneously dense tissue potentially obscuring small masses
- **Category 4:** Extremely dense tissue

![ACR standardized breast density](/images/breast-cancer-diagnosis/density.png)
*Figure 8: ACR standardized breast density. Reproduced from Mračko et al.*

## References

1. **World Health Organization**. "Breast cancer." 2024. [https://www.who.int/news-room/fact-sheets/detail/breast-cancer](https://www.who.int/news-room/fact-sheets/detail/breast-cancer)

2. **American Cancer Society**. "Survival Rates for Breast Cancer." 2025. [https://www.cancer.org/cancer/types/breast-cancer/understanding-a-breast-cancer-diagnosis/breast-cancer-survival-rates.html](https://www.cancer.org/cancer/types/breast-cancer/understanding-a-breast-cancer-diagnosis/breast-cancer-survival-rates.html)

3. **Alghaib, H. A., Scott, M., & Adhami, R. R.** (2016). "An Overview of Mammogram Analysis." *IEEE Potentials*, 35(6), 21-28. DOI: [10.1109/MPOT.2015.2396533](https://doi.org/10.1109/MPOT.2015.2396533)

4. **Mračko, A., Vanovčanová, L., & Cimrák, I.** (2023). "Mammography Datasets for Neural Networks—Survey." *Journal of Imaging*, 9(5), 95. DOI: [10.3390/jimaging9050095](https://www.mdpi.com/2313-433X/9/5/95)

5. **Liu, Y., Zhang, F., Chen, C., Wang, S., Wang, Y., & Yu, Y.** (2022). "Act Like a Radiologist: Towards Reliable Multi-View Correspondence Reasoning for Mammogram Mass Detection." *IEEE Transactions on Pattern Analysis and Machine Intelligence*, 44(10), 5947-5961. DOI: [10.1109/TPAMI.2021.3085783](https://doi.org/10.1109/TPAMI.2021.3085783)

6. **Kontos, D., Bakic, P., & Maidment, A.** (2008). "Texture in digital breast tomosynthesis: A comparison between mammographic and tomographic characterization of parenchymal properties." *Progress in Biomedical Optics and Imaging - Proceedings of SPIE*, 6915. DOI: [10.1117/12.773144](https://www.imagephysics.com/088.pdf)

7. **Joines, M. M., Dubin, I., & Mortazavi, S.** (2022). "Breast MRI." In *Absolute Breast Imaging Review: Multimodality Cases for the Core Exam* (pp. 193-239). Springer International Publishing. DOI: [10.1007/978-3-031-08274-0_5](https://doi.org/10.1007/978-3-031-08274-0_5)

8. **Radiology Cafe**. "X-ray imaging." 2024. [https://www.radiologycafe.com/frcr-physics-notes/x-ray-imaging/](https://www.radiologycafe.com/frcr-physics-notes/x-ray-imaging/)

9. **Johns Hopkins Pathology**. "Overview of the Breast." 2024. [https://pathology.jhu.edu/breast/overview](https://pathology.jhu.edu/breast/overview)

10. **American College of Radiology**. "ACR Breast Imaging Reporting & Data System (BI-RADS®)." 5th ed. Reston, VA: American College of Radiology, 2013. [https://www.acr.org/Clinical-Resources/Clinical-Tools-and-Reference/Reporting-and-Data-Systems/BI-RADS](https://www.acr.org/Clinical-Resources/Clinical-Tools-and-Reference/Reporting-and-Data-Systems/BI-RADS)

11. **Parikh, J.** (2024). "Burnout Fueling Workforce Woes." *ACR Bulletin*, American College of Radiology. [https://www.acr.org/Practice-Management-Quality-Informatics/ACR-Bulletin/Articles/July-2024/Burnout-Fueling-Workforce-Woes](https://www.acr.org/Practice-Management-Quality-Informatics/ACR-Bulletin/Articles/July-2024/Burnout-Fueling-Workforce-Woes)

12. **Kasalak, Ö., Alnahwi, H., Toxopeus, R., Pennings, J. P., Yakar, D., & Kwee, T. C.** (2023). "Work overload and diagnostic errors in radiology." *European Journal of Radiology*, 167, 111032. DOI: [10.1016/j.ejrad.2023.111032](https://doi.org/10.1016/j.ejrad.2023.111032)

13. **Zonderland, H. M., Smithuis, F., van Straten, M., et al.** (2020). "Workload for radiologists during on-call hours: dramatic increase in the past 15 years." *Insights into Imaging*, 11(1), 1-8. DOI: [10.1186/s13244-020-00925-z](https://doi.org/10.1186/s13244-020-00925-z)

---

*This blog post is part of a series on medical imaging and computer-aided diagnosis in breast cancer detection.*