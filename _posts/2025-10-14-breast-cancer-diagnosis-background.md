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

<style>
.highlight-box {
  background: linear-gradient(135deg, rgba(14, 161, 197, 0.1), rgba(14, 161, 197, 0.05));
  border-left: 4px solid #0ea1c5;
  padding: 20px;
  margin: 20px 0;
  border-radius: 8px;
  box-shadow: 0 2px 10px rgba(14, 161, 197, 0.1);
}

.section-header {
  color: #0ea1c5;
  border-bottom: 3px solid #0ea1c5;
  padding-bottom: 10px;
  margin-bottom: 25px;
  font-weight: bold;
}

.subsection-header {
  color: #0ea1c5;
  border-left: 4px solid #0ea1c5;
  padding-left: 15px;
  margin: 25px 0 15px 0;
  font-weight: 600;
}

.key-stat {
  color: #0ea1c5;
  font-weight: bold;
  font-size: 1.1em;
}

.figure-caption {
  text-align: center;
  font-style: italic;
  color: #666;
  margin-top: 10px;
  margin-bottom: 25px;
}

.definition-box {
  background: rgba(14, 161, 197, 0.05);
  border: 1px solid rgba(14, 161, 197, 0.2);
  border-radius: 6px;
  padding: 15px;
  margin: 15px 0;
}

.calcification-list {
  background: rgba(14, 161, 197, 0.05);
  border-radius: 8px;
  padding: 20px;
  margin: 20px 0;
}

.calcification-list li {
  margin: 8px 0;
  padding: 5px;
}

.responsive-figure-container {
  display: flex;
  justify-content: space-around;
  align-items: center;
  margin: 20px 0;
  flex-wrap: wrap;
}

.responsive-figure-container img {
  margin: 10px;
  max-width: 350px;
  width: 100%;
  height: auto;
}

@media (max-width: 768px) {
  .responsive-figure-container {
    flex-direction: column;
    align-items: center;
  }
  
  .responsive-figure-container img {
    max-width: 90%;
    margin: 10px 0;
  }
}
</style>

<div class="highlight-box">
<strong>Overview:</strong> This comprehensive guide introduces the core challenges in breast cancer detection, exploring the medical context, imaging technologies, and the critical need for computer-aided diagnosis systems.
</div>

<h2 class="section-header">Medical Context</h2>

Breast cancer is one of the most common oncological diseases in the world with <span class="key-stat">2.3 million new cases identified each year</span>. It arises either from hereditary genetic factors or lifestyle practices. Anatomically, breast cancer is caused by the uncontrolled multiplication of breast cells. Depending on the biological behavior of these cells, there are two forms of tumor: benign and malignant (Figure 1). Benign tumors are formed by cells that partially retain their morphology and function, although they continue to multiply. Malignant tumors, also called cancers, on the other hand, are characterized by a morphology and function that are different from those associated with healthy tissues. During their reproduction, they form extensions that infiltrate adjacent tissues, enveloping normal cells and destroying them, a phenomenon known as neoplastic invasiveness.

<img src="/images/breast-cancer-diagnosis/mass.png" alt="Examples of benign and malignant masses" width="600" style="display: block; margin: 0 auto;"/>
<div class="figure-caption">Figure 1: Examples of benign and malignant masses. Reproduced from Alghaib et al.</div>

<div class="highlight-box">
<strong style="color: #0ea1c5;">Key Insight:</strong> Early diagnosis on screening programs is the most effective tool for reducing mortality associated with neoplasms
</div>

Cancer screening involves conducting surveillance tests on a person who is supposedly healthy, with the aim of detecting abnormalities that could be warning signs of cancer, well before the first symptoms appear. This approach has proven highly effective in that it significantly reduces the mortality rate from breast cancer by improving the chances of recovery.. Indeed, if detected early, breast cancer can be cured in <span class="key-stat">nine out of ten cases</span>, contributing to a <span class="key-stat">15 to 21% reduction</span> in the mortality rate from this cancer.

<h2 class="section-header">Mammography</h2>

<h3 class="subsection-header">Presentation</h3>

The gold standard in screening programs is represented by mammography, that is, a precise and reliable diagnostic test, capable of detecting nodular lesions, even small ones, not yet detectable by touch. Mammography is also able to detect the presence of microcalcifications (small calcium deposits due to the secretions of mutated cells) that can be an indication of precancerous lesions. Distribution patterns of calcifications can be:

<div class="calcification-list">
<strong style="color: #0ea1c5; margin-bottom: 10px; display: block;">Calcification Distribution Patterns:</strong>
<ul style="list-style: none; padding-left: 0;">
<li><strong style="color: #0ea1c5;">• Diffuse:</strong> Random distribution throughout breast tissue → <span style="color: #28a745;">benign</span></li>
<li><strong style="color: #0ea1c5;">• Regional:</strong> Significant tissue proportion (>2cm) → <span style="color: #28a745;">likely benign</span></li>
<li><strong style="color: #0ea1c5;">• Cluster:</strong> Five or more calcifications in 1-2cm area → <span style="color: #ffc107;">malignancy risk</span></li>
<li><strong style="color: #0ea1c5;">• Linear:</strong> Arranged in ductal pattern → <span style="color: #fd7e14;">potential malignancy</span></li>
<li><strong style="color: #0ea1c5;">• Segmental:</strong> Deposits in duct systems and branches → <span style="color: #dc3545;">malignancy</span></li>
</ul>
</div>

<img src="/images/breast-cancer-diagnosis/calcifications.png" alt="Distribution of calcifications" width="700" style="display: block; margin: 0 auto;"/>
<div class="figure-caption">Figure 2: Distribution of calcifications. Reproduced from Mračko et al.</div>

Mammography employs an X-ray beam using an X-ray tube, oriented so as to be tangent to the patient's sternum (see Fig. 3). X-rays are radiations capable of penetrating inside biological tissues and reaching the detector. The quantity of radiation absorbed by the body depends on the type of object being examined and in particular it is greater the denser the body crossed is. Mammographs use two compression plates to distribute breast tissue so as to optimize image quality.

<h3 class="subsection-header">X-ray Beamforming</h3>

When high-speed electrons strike a metal target, their kinetic energy is transformed into heat (99%), and into X-rays (1%). An X-ray tube consists of three elements: an electron source (cathode), a potential difference, and a metal target for generating X-rays (anode). The assembly is contained within a vacuum chamber and a leaded sheath containing a window that allows the X-ray beam to pass through.

<div class="responsive-figure-container">
  <img src="/images/breast-cancer-diagnosis/mammo2.png" alt="X-ray tube design"/>
  <img src="/images/breast-cancer-diagnosis/mammo.png" alt="Mammograph design"/>
</div>
<div class="figure-caption">Figure 3: (a) Illustration of X-ray tube design (b) Illustration of mammograph design. Adapted from Radiology Cafe</div>

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

<img src="/images/breast-cancer-diagnosis/ccmlo.png" alt="Relations between CC and MLO views" width="500" style="display: block; margin: 0 auto;"/>
<div class="figure-caption">Figure 4: Relations between CC and MLO views. Reproduced from Liu et al.</div>

<h2 class="section-header">Other Imaging Tools</h2>

Tomosynthesis, also called Digital Breast Tomosynthesis (DBT), is a sophisticated mammography method in which the X-ray tube rotates in an arc across the breast to take several low-dose pictures from different angles usually ±25° (see Fig. 5). Compared to conventional 2D mammography, studies show higher cancer detection rates in DBT thanks to its 3D layered images which reduce tissue overlap.

<img src="/images/breast-cancer-diagnosis/tomo.png" alt="Illustration of Tomosynthesis" width="450" style="display: block; margin: 0 auto;"/>
<div class="figure-caption">Figure 5: Illustration of Tomosynthesis. Reproduced from Kontos et al.</div>

Magnetic Resonance Imaging uses radio waves and magnetic fields to create a detailed picture of breast tissue. It uses a large, tube-shaped equipment that produces intense magnetic fields for detection (see Fig. 6). The main drawback is the rise in false-positive rates, which may call for needless follow-up testing and raise patient worry and medical expenses. Hence, MRI is often employed as an additional screening technique for high-risk patients.

<div class="responsive-figure-container">
  <img src="/images/breast-cancer-diagnosis/mri.png" alt="MRI tube design"/>
  <img src="/images/breast-cancer-diagnosis/mri2.png" alt="Breast MRI scan"/>
</div>
<div class="figure-caption">Figure 6: (a) Illustration of MRI tube design (b) Breast MRI scan. Sources: (a) reproduced from Radiology Cafe, (b) reproduced from Joines et al.</div>

Despite the enhanced diagnostic capabilities of MRI and tomosynthesis, scalable deployment across healthcare systems is restricted by <span class="key-stat">high costs and restricted accessibility</span>. These technologies restrict throughput in high-volume screening settings since they call for specific tools, skilled workers, and lengthy examination periods. Mammography is therefore a commonly used method for diagnosing breast cancer because it balances diagnostic accuracy with operational feasibility.

<h2 class="section-header">Need for CAD Systems</h2>

Mammographic interpretation presents significant challenges in clinical practice that have motivated the development of Computer-Aided Detection (CAD) systems. These challenges stem from both the inherent complexity of mammographic images and systematic healthcare delivery issues.

<h3 class="subsection-header">Workload of Radiologists and Resource Limitations</h3>

There is a severe lack of skilled radiologists in the world's healthcare system which highly influences the precision of diagnosis. The Association of American Medical Colleges projects a shortage of 17,000 to 42,000 radiologists, pathologists, and psychiatrists by 2033. Current indicators of this shortage include over 1,400 open radiologist positions on the American College of Radiology's job board and reports that 53% of practicing radiologists are age 55 or older. Research indicates that radiologists who work above their usual capacity encounter significantly higher rates of diagnostic mistakes; workload normalized scores on days when diagnostic errors occurred average 121% higher than baseline levels.

Radiologist workload has skyrocketed in recent years; according to one European research, on-call workload doubled over a 15-year period in terms of relative value units. In screening programs, when a high volume of primarily normal patients must be handled quickly, this workload strain is especially severe.

<h3 class="subsection-header">Breast Density Problem</h3>

Macroscopically each breast has three main components: the mammary gland, the skin and the nipple-areola complex (see Fig. 7). The gland is surrounded by a dense network of connective tissue including arteries, veins, nerves and lymphatic vessels.

<img src="/images/breast-cancer-diagnosis/breast.png" alt="Breast anatomy illustration" width="400" style="display: block; margin: 0 auto;"/>
<div class="figure-caption">Figure 7: Breast anatomy illustration. Reproduced from Johns Hopkins Pathology</div>

Similar to cancers, dense breast tissue appears white on mammograms. Because of this, it becomes difficult to distinguish between normal and cancerous tissue in dense breasts using mammography.

ACR BI-RADS Atlas 2013 standardization framework utilizes categories a-d based on fibroglandular tissue, also called dense tissue, likelihood to obscure masses:

<div class="definition-box">
<strong style="color: #0ea1c5; display: block; margin-bottom: 15px;">ACR BI-RADS Breast Density Categories:</strong>
<ul style="list-style: none; padding-left: 0;">
<li style="margin: 10px 0; padding: 8px; background: rgba(14, 161, 197, 0.05); border-radius: 4px;"><strong style="color: #0ea1c5;">Category 1:</strong> Almost entirely fatty breasts</li>
<li style="margin: 10px 0; padding: 8px; background: rgba(14, 161, 197, 0.08); border-radius: 4px;"><strong style="color: #0ea1c5;">Category 2:</strong> Scattered fibroglandular density areas</li>
<li style="margin: 10px 0; padding: 8px; background: rgba(14, 161, 197, 0.12); border-radius: 4px;"><strong style="color: #0ea1c5;">Category 3:</strong> Heterogeneously dense tissue potentially obscuring small masses</li>
<li style="margin: 10px 0; padding: 8px; background: rgba(14, 161, 197, 0.15); border-radius: 4px;"><strong style="color: #0ea1c5;">Category 4:</strong> Extremely dense tissue</li>
</ul>
</div>

<img src="/images/breast-cancer-diagnosis/density.png" alt="ACR standardized breast density" width="550" style="display: block; margin: 0 auto;"/>
<div class="figure-caption">Figure 8: ACR standardized breast density. Reproduced from Mračko et al.</div>

<h2 class="section-header">References</h2>

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