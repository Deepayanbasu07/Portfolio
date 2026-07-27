import { useState } from 'react';
import IonIcon from './IonIcon';

export default function Sidebar() {
  const [isOpen, setIsOpen] = useState(false);

  return (
    <aside className={`sidebar${isOpen ? ' active' : ''}`}>
      <div className="sidebar-info">
        <figure className="avatar-box">
          <img src="/assets/deepu/dp.jpeg" alt="Deepayan Basu" width="80" />
        </figure>

        <div className="info-content">
          <h1 className="name" title="Deepayan Basu">Deepayan Basu</h1>
          <p className="title">Associate Consultant/Executive M&V Quant @KPMG</p>
          <p className="title">Data Scientist</p>
        </div>

        <button
          type="button"
          className="info_more-btn"
          onClick={() => setIsOpen((prev) => !prev)}
        >
          <span>Show Contacts</span>
          <IonIcon name="chevron-down" />
        </button>
      </div>

      <div className="sidebar-info_more">
        <div className="separator" />

        <ul className="contacts-list">
          <li className="contact-item">
            <div className="icon-box">
              <IonIcon name="mail-outline" />
            </div>
            <div className="contact-info">
              <p className="contact-title">Email</p>
              <a href="mailto:deepayanbasu5@gmail.com" className="contact-link">
                deepayanbasu5@gmail.com
              </a>
            </div>
          </li>

          <li className="contact-item">
            <div className="icon-box">
              <IonIcon name="phone-portrait-outline" />
            </div>
            <div className="contact-info">
              <p className="contact-title">Phone</p>
              <a href="tel:+918100537113" className="contact-link">
                +91 81005 37113
              </a>
            </div>
          </li>

          <li className="contact-item">
            <div className="icon-box">
              <IonIcon name="location-outline" />
            </div>
            <div className="contact-info">
              <p className="contact-title">Location</p>
              <address>Bengaluru, India</address>
            </div>
          </li>
        </ul>

        <div className="separator" />

        <ul className="social-list">
          <li className="social-item">
            <a href="https://github.com/Deepayanbasu07" target="_blank" rel="noreferrer" className="social-link">
              <IonIcon name="logo-github" />
            </a>
          </li>
          <li className="social-item">
            <a href="https://linkedin.com/in/deepayan-basu-06a5b123b" target="_blank" rel="noreferrer" className="social-link">
              <IonIcon name="logo-linkedin" />
            </a>
          </li>
          <li className="social-item">
            <a href="https://www.youtube.com/watch?v=KO5vkcHuggI" target="_blank" rel="noreferrer" className="social-link">
              <IonIcon name="logo-youtube" />
            </a>
          </li>
        </ul>
      </div>
    </aside>
  );
}
