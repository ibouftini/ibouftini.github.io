---
layout: single
permalink: /cv/
author_profile: false
classes: wide cv-page
redirect_from:
  - /resume
---

<link rel="stylesheet" href="{{ '/assets/css/cv.css' | relative_url }}">

<div class="cv-header">
  <div class="cv-profile-section">
    <div class="cv-profile-image">
      <img src="{{ '/images/profile.png' | relative_url }}" alt="Imade Bouftini" />
    </div>
    <div class="cv-profile-info">
      <h1 class="cv-name">Imade Bouftini</h1>
      <p class="cv-title">Generalist Engineering Student | AI Research Engineer</p>
      <div class="cv-contact">
        <div class="contact-item">
          <i class="fas fa-envelope"></i>
          <a href="mailto:imadebouftini@gmail.com">imadebouftini@gmail.com</a>
        </div>
        <div class="contact-item">
          <i class="fas fa-phone"></i>
          <a href="tel:+33778041481">+33 7 78 04 14 81</a>
        </div>
        <div class="contact-item">
          <i class="fas fa-map-marker-alt"></i>
          <span>Nantes, France</span>
        </div>
      </div>
      <div class="cv-social">
        <a href="https://linkedin.com/in/imade-bouftini" target="_blank" class="social-link">
          <i class="fab fa-linkedin"></i>
        </a>
        <a href="https://github.com/ibouftini" target="_blank" class="social-link">
          <i class="fab fa-github"></i>
        </a>
      </div>
    </div>
  </div>
</div>

<div class="cv-section">
  <h2><i class="fas fa-graduation-cap"></i> Education</h2>
  
  <div class="cv-entry">
    <div class="cv-entry-header">
      <h3 class="cv-entry-title">Generalist Engineering Degree</h3>
      <span class="cv-entry-date">Sept 2023 – 2026</span>
    </div>
    <p class="cv-entry-subtitle">École Centrale de Nantes</p>
    <p class="cv-entry-location">Nantes, France</p>
    <div class="cv-entry-description">
      <p><strong>1st Option:</strong> Data Science & Signal Processing | <strong>GPA:</strong> 3.98/4.0</p>
      <p><strong>Relevant courses:</strong> Optimization, Machine Learning, Deep Learning, Computer Vision, Signal Processing, Bayesian Statistics, Graph Inference</p>
      
      <p><strong>2nd Option:</strong> Mathematics & Applications | In progress</p>
      <p><strong>Relevant courses:</strong> NLP, Reinforcement Learning, Functional Analysis, Stochastic Processes, Numerical Methods, Uncertainty Quantification</p>
    </div>
  </div>
</div>

<div class="cv-section">
  <h2><i class="fas fa-briefcase"></i> Professional Experience</h2>
  
  <div class="cv-entry clickable" onclick="window.location.href='/portfolio/ai-movement-internship/'">
    <div class="cv-entry-header">
      <h3 class="cv-entry-title">
        <a href="/portfolio/ai-movement-internship/">AI Research Intern</a>
      </h3>
      <span class="cv-entry-date">Apr 2025 – Aug 2025</span>
    </div>
    <p class="cv-entry-subtitle">AI Movement</p>
    <p class="cv-entry-location">Rabat, Morocco</p>
    <div class="cv-entry-description">
      <ul>
        <li>Implemented and improved the "Act Like a Radiologist" paper for multi-view breast cancer detection under limited data</li>
        <li>Achieved <strong>9.5% improvement</strong> in Recall@0.5FPI over single-view baselines such as MaskRCNN, DETR and YOLO</li>
        <li><strong>Tools:</strong> PyTorch, OpenCV, GNNs, Detectron2, W&B, SLURM</li>
      </ul>
    </div>
  </div>
</div>

<div class="cv-section">
  <h2><i class="fas fa-flask"></i> Research Projects</h2>
  
  <div class="cv-entry clickable" onclick="window.location.href='/portfolio/fuel-cell-health-estimation/'">
    <div class="cv-entry-header">
      <h3 class="cv-entry-title">
        <a href="/portfolio/fuel-cell-health-estimation/">AI-based Fuel Cell Health Estimation</a>
      </h3>
      <span class="cv-entry-date">2024 – Present</span>
    </div>
    <p class="cv-entry-subtitle">LS2N Laboratory</p>
    <div class="cv-entry-description">
      <ul>
        <li>Built mathematical fuel cell model for training</li>
        <li>Developed Kalman-based health estimator for real-time monitoring</li>
        <li>Created a Physics-Informed Network with <strong>24% RMSE improvement</strong> on Extended Kalman Filter</li>
        <li>Validated framework on experimental long-term data in different scenarios</li>
      </ul>
    </div>
  </div>

  <div class="cv-entry clickable" onclick="window.location.href='/portfolio/sample-weighting-class-imbalance/'">
    <div class="cv-entry-header">
      <h3 class="cv-entry-title">
        <a href="/portfolio/sample-weighting-class-imbalance/">Sample Weighting for Class Imbalance Handling</a>
      </h3>
      <span class="cv-entry-date">2024</span>
    </div>
    <p class="cv-entry-subtitle">Hera MI</p>
    <div class="cv-entry-description">
      <ul>
        <li><strong>Medical AI:</strong> Developed breast cancer classification model (ResNet22 + CBAM), achieving <strong>70.3% specificity@0.9 sensitivity</strong></li>
        <li><strong>Optimization:</strong> Implemented "AUC Reshaping" paper, achieving <strong>11% performance improvement</strong></li>
      </ul>
    </div>
  </div>

  <div class="cv-entry clickable" onclick="window.location.href='/portfolio/multi-agent-rl-partial-observability/'">
    <div class="cv-entry-header">
      <h3 class="cv-entry-title">
        <a href="/portfolio/multi-agent-rl-partial-observability/">Multi-Agent RL Under Partial Observability</a>
      </h3>
      <span class="cv-entry-date">2024</span>
    </div>
    <p class="cv-entry-subtitle">Personal Project</p>
    <div class="cv-entry-description">
      <ul>
        <li>Developed Learning-Informed Masking for cooperative multi-agent systems using a masked auto-encoder</li>
        <li>Demonstrated <strong>27% faster convergence</strong> on QMIX in StarCraft Multi-Agent Challenge (SMAC)</li>
      </ul>
    </div>
  </div>
</div>

<div class="cv-section">
  <h2><i class="fas fa-code"></i> Technical Skills</h2>
  
  <div class="cv-skills-grid">
    <div class="cv-skill-category">
      <h3>Programming Languages</h3>
      <div class="cv-tags">
        <span class="cv-tag">Python</span>
        <span class="cv-tag">MATLAB</span>
        <span class="cv-tag">R</span>
      </div>
    </div>
    
    <div class="cv-skill-category">
      <h3>AI/ML Frameworks</h3>
      <div class="cv-tags">
        <span class="cv-tag">PyTorch</span>
        <span class="cv-tag">TensorFlow</span>
        <span class="cv-tag">scikit-learn</span>
        <span class="cv-tag">Langchain</span>
        <span class="cv-tag">Langgraph</span>
      </div>
    </div>
    
    <div class="cv-skill-category">
      <h3>Specialized Tools</h3>
      <div class="cv-tags">
        <span class="cv-tag">OpenCV</span>
        <span class="cv-tag">W&B</span>
        <span class="cv-tag">SLURM</span>
        <span class="cv-tag">Detectron2</span>
      </div>
    </div>
    
    <div class="cv-skill-category">
      <h3>Data & Infrastructure</h3>
      <div class="cv-tags">
        <span class="cv-tag">PostgreSQL</span>
        <span class="cv-tag">Pandas</span>
        <span class="cv-tag">NumPy</span>
        <span class="cv-tag">Docker</span>
        <span class="cv-tag">Git</span>
        <span class="cv-tag">FastAPI</span>
        <span class="cv-tag">AWS SageMaker</span>
      </div>
    </div>
  </div>
</div>

<div class="cv-section">
  <h2><i class="fas fa-language"></i> Languages</h2>
  
  <div class="cv-languages">
    <div class="cv-language">
      <span class="cv-language-name">Arabic</span>
      <span class="cv-language-level">Native</span>
    </div>
    <div class="cv-language">
      <span class="cv-language-name">French</span>
      <span class="cv-language-level">Fluent</span>
    </div>
    <div class="cv-language">
      <span class="cv-language-name">English</span>
      <span class="cv-language-level">Advanced</span>
    </div>
    <div class="cv-language">
      <span class="cv-language-name">Chinese</span>
      <span class="cv-language-level">Notions</span>
    </div>
  </div>
</div>

<div class="cv-section">
  <h2><i class="fas fa-users"></i> Leadership & Volunteering</h2>
  
  <div class="cv-entry">
    <div class="cv-entry-header">
      <h3 class="cv-entry-title">Teaching Volunteer</h3>
      <span class="cv-entry-date">2024 – 2025</span>
    </div>
    <p class="cv-entry-subtitle">Lycée Livet</p>
    <p class="cv-entry-location">Nantes, France</p>
    <div class="cv-entry-description">
      <ul>
        <li>Tutored "Classes Préparatoires" students in scientific and technical subjects</li>
        <li>Developed effective communication strategies for complex mathematical and engineering concepts</li>
      </ul>
    </div>
  </div>

  <div class="cv-entry">
    <div class="cv-entry-header">
      <h3 class="cv-entry-title">President of Student Council</h3>
      <span class="cv-entry-date">2021 – 2023</span>
    </div>
    <p class="cv-entry-subtitle">Lycée Mohammed VI d'Excellence</p>
    <p class="cv-entry-location">Benguerir, Morocco</p>
    <div class="cv-entry-description">
      <ul>
        <li>Led team of <strong>15+ council members</strong> in organizing 10+ key campus events and initiatives</li>
        <li>Maintained effective communication channels with administration for student advocacy and policy improvements</li>
      </ul>
    </div>
  </div>
</div>

