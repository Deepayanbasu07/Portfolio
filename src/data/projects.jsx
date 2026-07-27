export const projectCategories = [
  'All',
  'GenAI & LLMs',
  'Data Science & ML',
  'Materials Informatics',
];

export const projects = [
  {
    id: 'llm-serving',
    title: 'LLM Serving Lab (vLLM)',
    category: 'GenAI & LLMs',
    categoryKey: 'genai & llms',
    image: '/assets/images/project_llm_serving.png',
    alt: 'High-Throughput LLM Serving Lab',
    details: (
      <>
        <p style={{ marginBottom: '15px' }}>
          <strong>Description:</strong> Architected an end-to-end benchmarking harness transitioning SmolLM-135M from Hugging Face inference to highly optimized vLLM topologies.
        </p>
        <p style={{ marginBottom: '15px' }}>
          <strong>Tech Stack:</strong> vLLM, SmolLM-135M, FastAPI, Gradio, Python, Docker
        </p>
        <p style={{ marginBottom: '10px' }}>
          <strong>Key Implementation & Achievements:</strong>
        </p>
        <ul style={{ listStyleType: 'disc', paddingLeft: '20px', marginBottom: '15px' }}>
          <li style={{ marginBottom: '5px' }}>Simulated PagedAttention vs. contiguous memory allocation, successfully recovering ~80% of KV-cache fragmentation.</li>
          <li style={{ marginBottom: '5px' }}>Boosted system token utilization up to ~95%.</li>
          <li style={{ marginBottom: '5px' }}>Built an OpenAI-compatible FastAPI server integrated with a live Gradio telemetry dashboard, load-tested to support 20 concurrent requests.</li>
        </ul>
      </>
    ),
  },
  {
    id: 'finbot',
    title: 'FinBot Financial QA',
    category: 'GenAI & LLMs',
    categoryKey: 'genai & llms',
    image: '/assets/images/project_finbot.png',
    alt: 'FinBot Document QA',
    details: (
      <>
        <p style={{ marginBottom: '15px' }}>
          <strong>Description:</strong> Engineered a PDF-based financial Question-Answering assistant (FinBot) utilizing advanced retrieval architectures and graph databases during LLM internship at Scientific Investing.
        </p>
        <p style={{ marginBottom: '15px' }}>
          <strong>Tech Stack:</strong> LangChain, PyMuPDF, FinBERT, ChromaDB, Neo4j, nomic-embed-text, MCP Server
        </p>
        <p style={{ marginBottom: '10px' }}>
          <strong>Key Implementation & Achievements:</strong>
        </p>
        <ul style={{ listStyleType: 'disc', paddingLeft: '20px', marginBottom: '15px' }}>
          <li style={{ marginBottom: '5px' }}>Achieved a 40%+ improvement in answer relevance using advanced RAG retrieval pipelines.</li>
          <li style={{ marginBottom: '5px' }}>Integrated Neo4j knowledge graphs for relational QA across 1,000+ financial entities, improving entity extraction and query accuracy by 30%.</li>
          <li style={{ marginBottom: '5px' }}>Deployed the system on an MCP server, resulting in 2x faster response times.</li>
        </ul>
      </>
    ),
  },
  {
    id: 'credit-risk',
    title: 'Credit Risk Dashboard',
    category: 'Data Science & ML',
    categoryKey: 'data science & ml',
    image: '/assets/images/project_credit_risk.png',
    alt: 'Credit Risk Analysis Dashboard',
    details: (
      <>
        <p style={{ marginBottom: '15px' }}>
          <strong>Description:</strong> Developed an Advanced Credit Risk Analysis Dashboard featuring multiple machine learning algorithms, model monitoring, and statistical auditing for retail and wholesale loan portfolios.
        </p>
        <p style={{ marginBottom: '15px' }}>
          <strong>Tech Stack:</strong> Python, Scikit-learn, PyCaret, Statsmodels, PD/LGD/EAD Frameworks, Streamlit, Plotly
        </p>
        <p style={{ marginBottom: '10px' }}>
          <strong>Key Implementation & Achievements:</strong>
        </p>
        <ul style={{ listStyleType: 'disc', paddingLeft: '20px', marginBottom: '15px' }}>
          <li style={{ marginBottom: '5px' }}>Integrated credit risk framework structures: Probability of Default (PD), Loss Given Default (LGD), and Exposure at Default (EAD).</li>
          <li style={{ marginBottom: '5px' }}>Enabled comprehensive model monitoring diagnostics (Population Stability Index - PSI, Gini coefficient, KS-statistics).</li>
          <li style={{ marginBottom: '5px' }}>Implemented fairness audits and SHAP feature explainability plots to evaluate feature importance and interpretability of model decisions.</li>
        </ul>
      </>
    ),
  },
  {
    id: 'sentiment',
    title: 'Sentiment Analysis Pipeline',
    category: 'Data Science & ML',
    categoryKey: 'data science & ml',
    image: '/assets/images/project_credit_risk.png',
    alt: 'Sentiment Analysis Tool',
    details: (
      <>
        <p style={{ marginBottom: '15px' }}>
          <strong>Description:</strong> Built an automated conference call sentiment analysis tool designed for financial technology and asset intelligence pipelines.
        </p>
        <p style={{ marginBottom: '15px' }}>
          <strong>Tech Stack:</strong> Python, PyPDF2, Pandas, NLTK, VADER, Afinn
        </p>
        <p style={{ marginBottom: '10px' }}>
          <strong>Key Implementation & Achievements:</strong>
        </p>
        <ul style={{ listStyleType: 'disc', paddingLeft: '20px', marginBottom: '15px' }}>
          <li style={{ marginBottom: '5px' }}>Parsed and analyzed 50+ financial earnings transcripts to extract textual statements and key context.</li>
          <li style={{ marginBottom: '5px' }}>Generated quarter-over-quarter sentiment vectors correlating sentiment shifts with company outlook changes.</li>
          <li style={{ marginBottom: '5px' }}>Achieved a 73% correlation between post-call sentiment scores and actual stock price movements.</li>
        </ul>
      </>
    ),
  },
  {
    id: 'causal-ml',
    title: 'Causal ML (Applied Econometrics)',
    category: 'Data Science & ML',
    categoryKey: 'data science & ml',
    image: '/assets/images/project_credit_risk.png',
    alt: 'Causal ML Travel App',
    details: (
      <>
        <p style={{ marginBottom: '15px' }}>
          <strong>Description:</strong> Conducted a causal econometrics analysis to estimate treatment effects of promotional strategies on a travel application&apos;s participation rates.
        </p>
        <p style={{ marginBottom: '15px' }}>
          <strong>Tech Stack:</strong> Python, R, Causal ML, Scikit-learn, Causal Forests, Pandas
        </p>
        <p style={{ marginBottom: '10px' }}>
          <strong>Key Implementation & Achievements:</strong>
        </p>
        <ul style={{ listStyleType: 'disc', paddingLeft: '20px', marginBottom: '15px' }}>
          <li style={{ marginBottom: '5px' }}>Applied Causal Trees and Causal Forests to estimate Conditional Average Treatment Effects (CATE) across a survey cohort of over 600 users.</li>
          <li style={{ marginBottom: '5px' }}>Discovered and isolated a highly reward-responsive customer segment (+3.36 CATE) concentrated within the 14–20 age demographic.</li>
          <li style={{ marginBottom: '5px' }}>Created actionable targeting strategies to optimize conversion rates.</li>
        </ul>
      </>
    ),
  },
  {
    id: 'perovskite',
    title: 'Perovskite Band Gap Prediction',
    category: 'Materials Informatics',
    categoryKey: 'materials informatics',
    image: '/assets/images/project_materials.png',
    alt: 'Perovskites Band Gap Prediction',
    details: (
      <>
        <p style={{ marginBottom: '15px' }}>
          <strong>Description:</strong> Undertook B.Tech Project under the guidance of Prof. Appala Naidu Gandi at IIT Jodhpur to accelerate materials discovery using machine learning.
        </p>
        <p style={{ marginBottom: '15px' }}>
          <strong>Tech Stack:</strong> Python, PyMatGen, Extra Trees, Random Forest Regressors, Materials Project API, VASP, VESTA, VASPKIT
        </p>
        <p style={{ marginBottom: '10px' }}>
          <strong>Key Implementation & Achievements:</strong>
        </p>
        <ul style={{ listStyleType: 'disc', paddingLeft: '20px', marginBottom: '15px' }}>
          <li style={{ marginBottom: '5px' }}>Extracted over 4,500 perovskite structures from the Materials Project Database using PyMatGen.</li>
          <li style={{ marginBottom: '5px' }}>Engineered 20+ physics-based descriptors (ionization potentials, atomic numbers, crystal bond lengths).</li>
          <li style={{ marginBottom: '5px' }}>Trained Extra Trees and Random Forest regressors to predict electronic band gaps and thermodynamic phase stability, replacing slow DFT calculations (R² = 0.83, F1 = 0.83).</li>
        </ul>
      </>
    ),
  },
  {
    id: 'lankford',
    title: 'Lankford Metallurgy CNN',
    category: 'Materials Informatics',
    categoryKey: 'materials informatics',
    image: '/assets/images/project_materials.png',
    alt: 'Lankford Coefficient Prediction',
    details: (
      <>
        <p style={{ marginBottom: '15px' }}>
          <strong>Description:</strong> Built a computer vision deep learning model to predict Lankford anisotropy coefficients directly from metallurgical Orientation Distribution Function (ODF) images.
        </p>
        <p style={{ marginBottom: '15px' }}>
          <strong>Tech Stack:</strong> Python, TensorFlow, Keras, OpenCV, Metallurgy ODF analytics
        </p>
        <p style={{ marginBottom: '10px' }}>
          <strong>Key Implementation & Achievements:</strong>
        </p>
        <ul style={{ listStyleType: 'disc', paddingLeft: '20px', marginBottom: '15px' }}>
          <li style={{ marginBottom: '5px' }}>Processed over 10,000 grain texture ODF segments through a custom CNN architecture.</li>
          <li style={{ marginBottom: '5px' }}>Achieved outstanding test evaluation scores: R² values of 0.89, 0.92, and 0.94 in principal anisotropy directions.</li>
          <li style={{ marginBottom: '5px' }}>Reduced metallurgical simulation processing time by over 80%.</li>
        </ul>
      </>
    ),
  },
  {
    id: 'sar',
    title: 'SAR Target Classification (DRDO)',
    category: 'Data Science & ML',
    categoryKey: 'data science & ml',
    image: '/assets/images/project_llm_serving.png',
    alt: 'SAR Target Classification',
    details: (
      <>
        <p style={{ marginBottom: '15px' }}>
          <strong>Description:</strong> Designed and implemented deep learning models for Synthetic Aperture Radar (SAR) target recognition during summer internship at Defence Laboratory, DRDO.
        </p>
        <p style={{ marginBottom: '15px' }}>
          <strong>Tech Stack:</strong> Python, TensorFlow, PyTorch, CNN, MSTAR Dataset, Task-Driven Pruning
        </p>
        <p style={{ marginBottom: '10px' }}>
          <strong>Key Implementation & Achievements:</strong>
        </p>
        <ul style={{ listStyleType: 'disc', paddingLeft: '20px', marginBottom: '15px' }}>
          <li style={{ marginBottom: '5px' }}>Architected a dual-branch CNN (J-CNN) specializing in high-noise target recognition.</li>
          <li style={{ marginBottom: '5px' }}>Achieved a peak classification accuracy of 94% on the military-grade MSTAR radar target dataset.</li>
          <li style={{ marginBottom: '5px' }}>Applied task-driven network pruning, reducing target inference computational latency by 23%.</li>
        </ul>
      </>
    ),
  },
  {
    id: 'ram',
    title: 'Radar Absorbing Materials (DRDO)',
    category: 'Materials Informatics',
    categoryKey: 'materials informatics',
    image: '/assets/images/project_materials.png',
    alt: 'Radar Absorbing Materials',
    details: (
      <>
        <p style={{ marginBottom: '15px' }}>
          <strong>Description:</strong> Developed advanced Radar Absorbing Materials (RAM) in the Camouflage Division of Defence Laboratory, DRDO, to reduce radar cross-sections of stealth aircraft.
        </p>
        <p style={{ marginBottom: '15px' }}>
          <strong>Tech Stack:</strong> Glass-Carbon Hybrid Veil, Epoxy Resin matrices, Hand-Layup processing, vacuum bagging, Anechoic Chamber testing
        </p>
        <p style={{ marginBottom: '10px' }}>
          <strong>Key Implementation & Achievements:</strong>
        </p>
        <ul style={{ listStyleType: 'disc', paddingLeft: '20px', marginBottom: '15px' }}>
          <li style={{ marginBottom: '5px' }}>Fabricated composites and cured them in a specialized vacuum bagging setup.</li>
          <li style={{ marginBottom: '5px' }}>Tested prototype reflectivity in a professional Anechoic Chamber across the 8–18 GHz frequency range.</li>
          <li style={{ marginBottom: '5px' }}>Achieved a 30% reduction in Radar Cross-Section (RCS) parameters, indicating high radar signature absorption.</li>
        </ul>
      </>
    ),
  },
];
