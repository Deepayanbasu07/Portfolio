import IonIcon from './IonIcon';

export default function ProjectModal({ project, isOpen, onClose }) {
  if (!project || !isOpen) return null;

  return (
    <div className="modal-container active">
      <div className="overlay active" onClick={onClose} role="presentation" />
      <section
        className="testimonials-modal project-modal"
        style={{ maxWidth: '750px', width: '90%', maxHeight: '85vh', overflowY: 'auto' }}
      >
        <button type="button" className="modal-close-btn" onClick={onClose}>
          <IonIcon name="close-outline" />
        </button>
        <div className="project-modal-banner" style={{ marginBottom: '20px' }}>
          <img
            src={project.image}
            alt={project.alt}
            style={{ width: '100%', borderRadius: '12px', height: '250px', objectFit: 'cover' }}
          />
        </div>
        <div className="modal-content">
          <h4
            className="h3 modal-title"
            style={{ fontSize: 'var(--fs-2)', marginBottom: '5px' }}
          >
            {project.title}
          </h4>
          <p
            className="project-modal-category"
            style={{
              color: 'var(--vegas-gold)',
              fontSize: 'var(--fs-6)',
              fontWeight: 'var(--fw-500)',
              marginBottom: '20px',
            }}
          >
            {project.category}
          </p>
          <div
            className="project-modal-text"
            style={{ color: 'var(--light-gray)', fontSize: 'var(--fs-6)', lineHeight: 1.8 }}
          >
            {project.details}
          </div>
        </div>
      </section>
    </div>
  );
}
