import IonIcon from './IonIcon';

const services = [
  {
    icon: '/assets/images/icon-dev.svg',
    iconAlt: 'risk modeling icon',
    title: 'Data Science & Risk Modeling',
    text: 'Validating retail and wholesale credit risk models (PD, LGD, EAD) at KPMG using statistical significance tests and machine learning.',
  },
  {
    icon: '/assets/images/icon-app.svg',
    iconAlt: 'genai icon',
    title: 'GenAI & LLM Engineering',
    text: 'Engineering RAG systems, integrating Neo4j knowledge graphs, and benchmarking high-throughput inference serving using vLLM.',
  },
  {
    icon: '/assets/images/icon-design.svg',
    iconAlt: 'materials science icon',
    title: 'Materials Informatics',
    text: 'Applying machine learning to predict electronic properties, band gaps, and thermodynamic stability of perovskite compounds.',
  },
  {
    iconType: 'ion',
    title: 'Music & Event Production',
    text: 'Managing productions, coordinating logistics, playing multiple instruments, and performing with the Sangam Music Society.',
  },
];

export default function About({ isActive }) {
  return (
    <article className={`about${isActive ? ' active' : ''}`} data-page="about">
      <header>
        <h2 className="h2 article-title">About me</h2>
      </header>

      <section className="about-text">
        <p>
          I&apos;m currently an Associate Consultant in Model Risk Management (MRM) at KPMG Global Services, and a B.Tech graduate in Materials Engineering from the Indian Institute of Technology, Jodhpur (Class of 2025).
          While my academic background lies in Materials Science, I&apos;ve actively explored the intersection of Machine Learning, Deep Learning, Quantitative Risk Modeling, and Generative AI.
        </p>
        <p>
          From optimizing LLM serving to developing Radar Absorbing Materials and credit risk analysis pipelines, I love working at the crossroads of innovation, data science, and high-impact software engineering. My interests also include full-stack development, and in my spare time, I&apos;m a passionate guitar player and mentor for the Sangam Music Society.
        </p>
      </section>

      <section className="service">
        <h3 className="h3 service-title">What i&apos;m doing</h3>

        <ul className="service-list">
          {services.map((service) => (
            <li key={service.title} className="service-item">
              <div className="service-icon-box">
                {service.iconType === 'ion' ? (
                  <IonIcon
                    name="musical-notes-outline"
                    style={{
                      fontSize: '40px',
                      color: 'var(--orange-yellow-crayola)',
                      margin: 'auto',
                      display: 'block',
                    }}
                  />
                ) : (
                  <img src={service.icon} alt={service.iconAlt} width="40" />
                )}
              </div>

              <div className="service-content-box">
                <h4 className="h4 service-item-title">{service.title}</h4>
                <p className="service-item-text">{service.text}</p>
              </div>
            </li>
          ))}
        </ul>
      </section>
    </article>
  );
}
