import IonIcon from './IonIcon';

export default function Highlights({ isActive }) {
  return (
    <article className={`blog${isActive ? ' active' : ''}`} data-page="highlights">
      <header>
        <h2 className="h2 article-title">Highlights</h2>
      </header>

      <section className="highlights-content">
        <div className="highlight-card">
          <h3 className="h3 highlight-title">Academic Experience at IIT Jodhpur</h3>
          <div className="highlight-flex">
            <div className="highlight-text-col">
              <p>
                I&apos;m currently pursuing my B.Tech in Materials Science and Metallurgical Engineering at IIT Jodhpur. Over the years, I&apos;ve developed strong foundations in physical metallurgy, materials characterization, and computational simulations. I&apos;ve explored interdisciplinary fields like machine learning for material property prediction (band gaps, stress hotspots, etc.), with tools like VASP, LAMMPS, and Python-based data pipelines.
              </p>
              <p>
                My coursework included: Phase Transformations, Smart Materials, Material Processing, and ML for Materials. These helped me build a well-rounded understanding of both traditional metallurgical concepts and modern computational tools.
              </p>
            </div>
            <div className="highlight-img-col">
              <img src="/assets/deepu/college.webp" alt="IIT Jodhpur Campus" className="highlight-img" />
            </div>
          </div>
        </div>

        <div className="highlight-card">
          <h3 className="h3 highlight-title">B.Tech Project: Perovskite Band Gap & Stability</h3>
          <div className="highlight-flex reverse">
            <div className="highlight-text-col">
              <p>
                My B.Tech project, under the guidance of Prof. Appala Naidu Gandi at IIT Jodhpur, aimed to accelerate materials discovery using machine learning models that predict electronic properties—specifically the band gap and thermodynamic phase stability—of perovskite compounds.
              </p>
              <p>
                We extracted over 4,500 perovskite structures from the Materials Project Database using PyMatGen and derived descriptors such as average atomic numbers, ionization potentials, and bond lengths. These were used to train regression and classification models, replacing time-intensive DFT calculations.
              </p>
              <p>
                For band gap prediction, we used Extra Trees and Random Forest Regressors. For phase stability, we employed both regression (R² = 0.83) and classification (F1 = 0.83), determining whether a material is stable based on its energy above the convex hull (Ehull). We validated our predictions against actual DFT results and implemented workflows using VASP, VESTA, and VASPKIT.
              </p>
            </div>
            <div className="highlight-img-col">
              <img src="/assets/images/project_materials.png" alt="Crystal structure prediction" className="highlight-img" />
            </div>
          </div>
        </div>

        <div className="highlight-card">
          <h3 className="h3 highlight-title">Internship at DRDO – Defence Laboratory, Jodhpur</h3>
          <div className="highlight-flex">
            <div className="highlight-text-col">
              <p>
                During my Summer 2024 internship at DRDO (Ministry of Defence, Govt. of India), I worked in the Camouflage Division to develop advanced Radar Absorbing Materials (RAM) aimed at reducing the radar cross-section (RCS) of fighter aircraft. We used a classified Glass-Carbon Hybrid veil and Epoxy to fabricate samples using the Hand-Layup technique, followed by curing in a Vacuum Bagging system.
              </p>
              <p>
                I tested the prototypes in an Anechoic Chamber across the 8–18 GHz frequency band to analyze radiation losses. Further testing included exposure to Rain Test Chamber, Thermal Shock Chamber, and CHNSO Elemental Chamber for environmental durability.
              </p>
              <p>
                Alongside material research, I developed a joint despeckling–recognition CNN for SAR Target Classification. Using the MSTAR dataset and Task-Driven Pruning, the model achieved 94% accuracy while reducing inference time by 23%. This work bridged radar image processing with efficient deep learning.
              </p>
            </div>
            <div className="highlight-img-col">
              <img src="/assets/images/project_llm_serving.png" alt="Radar Research at DRDO" className="highlight-img" />
            </div>
          </div>
        </div>

        <div className="highlight-card">
          <h3 className="h3 highlight-title">Mentorship & Music</h3>
          <div className="highlight-flex reverse">
            <div className="highlight-text-col">
              <p>
                <strong>Mentor and Core Team Member of Sangam Music Society, IITJ:</strong><br />
                Handled multiple instruments and demonstrated proficiency in managing the society&apos;s diverse performances. Oversaw coordination between band members, sound engineers, and venue staff to ensure smooth and impactful shows.
              </p>
              <p style={{ marginTop: '20px' }}>
                <a
                  href="https://www.youtube.com/watch?v=KO5vkcHuggI"
                  target="_blank"
                  rel="noreferrer"
                  className="youtube-link"
                >
                  <IonIcon name="logo-youtube" />
                  <span>Watch My Music Video on YouTube</span>
                </a>
              </p>
            </div>
            <div className="highlight-img-col">
              <img src="/assets/deepu/dp3.jpg" alt="Deepayan playing guitar" className="highlight-img" />
            </div>
          </div>
        </div>

        <div className="highlight-card">
          <h3 className="h3 highlight-title">Interests and Aspirations</h3>
          <div className="highlight-flex">
            <div className="highlight-text-col">
              <p>
                I&apos;m passionate about AI, NLP, LLMs, and using data science to solve real-world business problems. I recently built FinBot, a chatbot that analyzes financial documents, and developed an Advanced Credit Risk Analysis Dashboard featuring multiple ML algorithms, model monitoring, and fairness audits.
              </p>
              <p>
                I&apos;m learning full-stack development to deploy my solutions at scale. My long-term goal is to bridge the gap between technical research and industry needs through practical, intelligent systems.
              </p>
            </div>
            <div className="highlight-img-col">
              <img src="/assets/deepu/dp.jpeg" alt="Deepayan Basu Activity" className="highlight-img" />
            </div>
          </div>
        </div>
      </section>
    </article>
  );
}
