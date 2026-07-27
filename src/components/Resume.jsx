import IonIcon from './IonIcon';

const education = [
  {
    title: 'Indian Institute of Technology, Jodhpur',
    period: 'Jul 2021 — May 2025',
    text: 'B.Tech. in Materials Engineering. Coursework: Physical Metallurgy, Phase Transformations, Smart Materials, Material Processing, and Machine Learning for Materials.',
  },
  {
    title: 'South Point High School (CBSE)',
    period: 'Apr 2019 — Mar 2021',
    text: 'Senior Secondary / Class XII. Completed with an aggregate of 94% (Kolkata, India).',
  },
  {
    title: 'South Point High School (CBSE)',
    period: 'Apr 2017 — Mar 2019',
    text: 'Secondary / Class X. Completed with an aggregate of 96% (Kolkata, India).',
  },
];

const experience = [
  {
    title: 'Associate Consultant – Model Risk Management (MRM)',
    period: 'KPMG Global Services | Nov 2025 — Present',
    text: 'Validated retail and wholesale PD, LGD, EAD credit risk models (Markov Chain, Dynamic Transition Matrix, Cox Proportional Hazard, Super Panel Hazard). Applied SHAP and Gini diagnostics for model explainability. Conducted quantitative performance testing using KS-statistic, PSI, and CSI.',
  },
  {
    title: 'LLM Intern',
    period: 'Scientific Investing | Nov 2024 — Feb 2025',
    text: 'Engineered a PDF-based financial QA system using LangChain, FinBERT, and ChromaDB, achieving 40%+ improvement in answer relevance. Integrated Neo4j knowledge graphs for relational QA across 1,000+ financial entities.',
  },
  {
    title: 'Research & Development Intern',
    period: 'DRDO, Defence Laboratory | May 2024 — Jul 2024',
    text: 'Designed a dual-branch CNN with task-driven pruning for SAR target classification, achieving 94% accuracy on MSTAR dataset. Modeled radar absorbing composites (RAM) for stealth aircraft, achieving a 30% reduction in radar cross-section (RCS).',
  },
];

const skills = [
  { name: 'Data Science & Risk Modeling', value: 90 },
  { name: 'GenAI & LLM Engineering', value: 85 },
  { name: 'Machine Learning & Deep Learning', value: 85 },
  { name: 'MLOps & Infrastructure', value: 75 },
];

export default function Resume({ isActive }) {
  return (
    <article className={`resume${isActive ? ' active' : ''}`} data-page="resume">
      <header>
        <h2 className="h2 article-title">Resume</h2>
      </header>

      <section className="timeline">
        <div className="title-wrapper">
          <div className="icon-box">
            <IonIcon name="book-outline" />
          </div>
          <h3 className="h3">Education</h3>
        </div>

        <ol className="timeline-list">
          {education.map((item) => (
            <li key={`${item.title}-${item.period}`} className="timeline-item">
              <h4 className="h4 timeline-item-title">{item.title}</h4>
              <span>{item.period}</span>
              <p className="timeline-text">{item.text}</p>
            </li>
          ))}
        </ol>
      </section>

      <section className="timeline">
        <div className="title-wrapper">
          <div className="icon-box">
            <IonIcon name="briefcase-outline" />
          </div>
          <h3 className="h3">Experience</h3>
        </div>

        <ol className="timeline-list">
          {experience.map((item) => (
            <li key={`${item.title}-${item.period}`} className="timeline-item">
              <h4 className="h4 timeline-item-title">{item.title}</h4>
              <span>{item.period}</span>
              <p className="timeline-text">{item.text}</p>
            </li>
          ))}
        </ol>
      </section>

      <section className="skill">
        <h3 className="h3 skills-title">My skills</h3>

        <ul className="skills-list content-card">
          {skills.map((skill) => (
            <li key={skill.name} className="skills-item">
              <div className="title-wrapper">
                <h5 className="h5">{skill.name}</h5>
                <data value={skill.value}>{skill.value}%</data>
              </div>

              <div className="skill-progress-bg">
                <div className="skill-progress-fill" style={{ width: `${skill.value}%` }} />
              </div>
            </li>
          ))}
        </ul>
      </section>
    </article>
  );
}
