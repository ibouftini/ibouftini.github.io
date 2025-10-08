---
permalink: /
author_profile: true
redirect_from: 
  - /about/
  - /about.html
---

<link rel="stylesheet" href="{{ '/assets/css/landing.css' | relative_url }}">

<div class="landing-hero">
  <div class="hero-content">
    <div class="hero-profile">
      <div class="profile-avatar">
        <img src="{{ '/images/profile.png' | relative_url }}" alt="Imade Bouftini">
      </div>
      <h1>Hello, I'm Imade Bouftini</h1>
      <p class="subtitle">Generalist Engineering Student</p>
      <p class="bio">AI Research Engineer | École Centrale de Nantes | Developing cutting-edge AI solutions for healthcare and industrial applications</p>
      <div class="location-info">
        <span class="location">
          <i class="fas fa-location-dot"></i>
          Nantes, France
        </span>
        <span class="employer">
          <i class="fas fa-building-columns"></i>
          École Centrale de Nantes
        </span>
      </div>
    </div>
    <div class="cta-buttons">
      <a href="/cv/" class="cta-button">
        <i class="fas fa-file-alt"></i>
        View My CV
      </a>
      <a href="/portfolio/" class="cta-button">
        <i class="fas fa-briefcase"></i>
        Explore My Work
      </a>
      <a href="#connect" class="cta-button cta-connect" onclick="scrollToConnect(event)">
        <i class="fas fa-handshake"></i>
        Let's Connect
      </a>
    </div>
  </div>
  <div class="scroll-indicator">
    <i class="fas fa-chevron-down"></i>
  </div>
</div>

<script>
function scrollToConnect(event) {
  event.preventDefault();
  document.querySelector('.landing-connect').scrollIntoView({
    behavior: 'smooth',
    block: 'start'
  });
}
</script>

<div class="landing-intro">
  <div class="intro-card">
    <h2>About Me</h2>
    <p>
      I'm a <span class="highlight">Generalist Engineering Student</span> at École Centrale de Nantes, 
      passionate about <span class="highlight">artificial intelligence</span> and its real-world applications. 
      My research focuses on developing innovative AI solutions for healthcare and industrial systems.
    </p>
    <p>
      This portfolio showcases my journey through cutting-edge research projects, academic achievements, 
      and the exploration of how AI can solve complex problems across different domains.
    </p>
  </div>
</div>

<div class="landing-portfolio">
  <h2>What You'll Find Here</h2>
  <div class="portfolio-grid">
    <div class="portfolio-card">
      <span class="icon">🎓</span>
      <h3>Academic Journey</h3>
      <p>My educational background, coursework, and academic achievements at École Centrale de Nantes.</p>
      <a href="/cv/" class="card-link">View Academic Background <i class="fas fa-arrow-right"></i></a>
    </div>
    
    <div class="portfolio-card">
      <span class="icon">🔬</span>
      <h3>Research Projects</h3>
      <p>Cutting-edge AI research in medical imaging, industrial monitoring, and multi-agent systems.</p>
      <a href="/portfolio/" class="card-link">Explore Research <i class="fas fa-arrow-right"></i></a>
    </div>
    
    <div class="portfolio-card">
      <span class="icon">💡</span>
      <h3>Blog & Insights</h3>
      <p>Thoughts on AI developments, technical tutorials, and insights from my research experience.</p>
      <a href="/year-archive/" class="card-link">Read Posts <i class="fas fa-arrow-right"></i></a>
    </div>
  </div>
</div>


<div class="landing-connect">
  <h2>Let's Connect</h2>
  <p class="connect-description">
    Interested in discussing AI research, potential collaborations, or just want to say hello? 
    I'd love to hear from you!
  </p>
  <div class="social-links">
    <a href="mailto:imadebouftini@gmail.com" class="social-link">
      <i class="fas fa-envelope"></i>
      Email
    </a>
    <a href="https://linkedin.com/in/imade-bouftini" class="social-link" target="_blank">
      <i class="fab fa-linkedin"></i>
      LinkedIn
    </a>
    <a href="https://github.com/ibouftini" class="social-link" target="_blank">
      <i class="fab fa-github"></i>
      GitHub
    </a>
  </div>
</div>