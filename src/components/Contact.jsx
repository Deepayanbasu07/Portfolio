import { useState } from 'react';
import IonIcon from './IonIcon';

export default function Contact({ isActive }) {
  const [formValid, setFormValid] = useState(false);

  const handleInput = (e) => {
    setFormValid(e.currentTarget.form.checkValidity());
  };

  return (
    <article className={`contact${isActive ? ' active' : ''}`} data-page="contact">
      <header>
        <h2 className="h2 article-title">Contact</h2>
      </header>

      <section
        className="resume-embed"
        style={{
          marginBottom: '40px',
          background: 'var(--border-gradient-onyx)',
          padding: '1px',
          borderRadius: '20px',
          boxShadow: 'var(--shadow-4)',
        }}
      >
        <div style={{ background: 'var(--eerie-black-2)', borderRadius: '19px', padding: '20px' }}>
          <div
            style={{
              display: 'flex',
              justifyContent: 'space-between',
              alignItems: 'center',
              marginBottom: '20px',
              flexWrap: 'wrap',
              gap: '15px',
            }}
          >
            <div style={{ display: 'flex', alignItems: 'center', gap: '12px' }}>
              <div
                className="icon-box"
                style={{
                  background: 'var(--onyx)',
                  color: 'var(--orange-yellow-crayola)',
                  width: '42px',
                  height: '42px',
                  borderRadius: '10px',
                  display: 'flex',
                  justifyContent: 'center',
                  alignItems: 'center',
                  fontSize: '20px',
                  boxShadow: 'var(--shadow-1)',
                  border: '1px solid var(--jet)',
                  marginBottom: 0,
                }}
              >
                <IonIcon name="document-text-outline" />
              </div>
              <div>
                <h3
                  className="h4"
                  style={{
                    margin: 0,
                    color: 'var(--white-2)',
                    fontWeight: 'var(--fw-600)',
                    fontSize: 'var(--fs-6)',
                  }}
                >
                  Resume_Deepayan_Basu.pdf
                </h3>
                <p style={{ margin: 0, color: 'var(--light-gray)', fontSize: '11px' }}>Updated July 2026</p>
              </div>
            </div>
            <a
              href="/assets/resume/Resume_Deepayan_Basu_July_2026.pdf"
              download
              className="form-btn"
              style={{
                width: 'max-content',
                padding: '8px 16px',
                fontSize: 'var(--fs-7)',
                display: 'inline-flex',
                alignItems: 'center',
                gap: '8px',
                margin: 0,
                borderRadius: '10px',
                background: 'var(--border-gradient-onyx)',
              }}
            >
              <IonIcon name="download-outline" style={{ fontSize: '16px' }} />
              <span>Download PDF</span>
            </a>
          </div>

          <div
            className="cv-iframe-container"
            style={{
              borderRadius: '12px',
              overflow: 'hidden',
              border: '1px solid var(--jet)',
              background: 'var(--eerie-black-1)',
              boxShadow: 'inset 0 2px 8px rgba(0, 0, 0, 0.5)',
              position: 'relative',
              paddingBottom: '141.4%',
              height: 0,
            }}
          >
            <iframe
              src="/assets/resume/Resume_Deepayan_Basu_July_2026.pdf"
              title="Resume Preview"
              style={{
                position: 'absolute',
                top: 0,
                left: 0,
                width: '100%',
                height: '100%',
                border: 'none',
              }}
            />
          </div>
        </div>
      </section>

      <section className="mapbox">
        <figure>
          <iframe
            src="https://www.google.com/maps/embed?pb=!1m18!1m12!1m3!1d3887.96766624467!2d77.5945627!3d12.9715987!2m3!1f0!2f0!3f0!3m2!1i1024!2i768!4f13.1!3m3!1m2!1s0x3bae1670c9b44e6d%3A0xf8dfc3e8517e4fe0!2sBengaluru%2C%20Karnataka%2C%20India!5e0!3m2!1sen!2sin!4v1680000000000!5m2!1sen!2sin"
            width="400"
            height="300"
            loading="lazy"
            title="Bengaluru Map"
          />
        </figure>
      </section>

      <section className="contact-form">
        <h3 className="h3 form-title">Contact Form</h3>

        <form
          action="https://formsubmit.co/deepayanbasu5@gmail.com"
          method="POST"
          className="form"
          onInput={handleInput}
        >
          <input type="hidden" name="_subject" value="New Portfolio Message!" />
          <input type="hidden" name="_honeypot" style={{ display: 'none' }} />
          <input type="hidden" name="_template" value="table" />

          <div className="input-wrapper">
            <input type="text" name="fullname" className="form-input" placeholder="Full name" required onInput={handleInput} />
            <input type="email" name="email" className="form-input" placeholder="Email address" required onInput={handleInput} />
          </div>

          <textarea name="message" className="form-input" placeholder="Your Message" required onInput={handleInput} />

          <button className="form-btn" type="submit" disabled={!formValid}>
            <IonIcon name="paper-plane" />
            <span>Send Message</span>
          </button>
        </form>
      </section>
    </article>
  );
}
