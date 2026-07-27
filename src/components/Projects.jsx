import { useState } from 'react';
import IonIcon from './IonIcon';
import { projectCategories, projects } from '../data/projects';

export default function Projects({ isActive, onProjectClick }) {
  const [activeFilter, setActiveFilter] = useState('All');
  const [selectOpen, setSelectOpen] = useState(false);

  const filterKey = activeFilter.toLowerCase();

  const isProjectVisible = (project) =>
    filterKey === 'all' || filterKey === project.categoryKey;

  const handleFilterChange = (category) => {
    setActiveFilter(category);
    setSelectOpen(false);
  };

  return (
    <article className={`portfolio${isActive ? ' active' : ''}`} data-page="projects">
      <header>
        <h2 className="h2 article-title">Projects</h2>
      </header>

      <section className="projects">
        <ul className="filter-list">
          {projectCategories.map((category) => (
            <li key={category} className="filter-item">
              <button
                type="button"
                className={activeFilter === category ? 'active' : ''}
                onClick={() => handleFilterChange(category)}
              >
                {category}
              </button>
            </li>
          ))}
        </ul>

        <div className="filter-select-box">
          <button
            type="button"
            className={`filter-select${selectOpen ? ' active' : ''}`}
            onClick={() => setSelectOpen((prev) => !prev)}
          >
            <div className="select-value">{activeFilter === 'All' ? 'Select category' : activeFilter}</div>
            <div className="select-icon">
              <IonIcon name="chevron-down" />
            </div>
          </button>

          <ul className="select-list">
            {projectCategories.map((category) => (
              <li key={category} className="select-item">
                <button type="button" onClick={() => handleFilterChange(category)}>
                  {category}
                </button>
              </li>
            ))}
          </ul>
        </div>

        <ul className="project-list">
          {projects.map((project) => (
            <li
              key={project.id}
              className={`project-item${isProjectVisible(project) ? ' active' : ''}`}
              data-category={project.categoryKey}
            >
              <a
                href="#"
                onClick={(e) => {
                  e.preventDefault();
                  onProjectClick(project);
                }}
              >
                <figure className="project-img">
                  <div className="project-item-icon-box">
                    <IonIcon name="eye-outline" />
                  </div>
                  <img src={project.image} alt={project.alt} loading="lazy" />
                </figure>

                <h3 className="project-title">{project.title}</h3>
                <p className="project-category">{project.category}</p>
              </a>
            </li>
          ))}
        </ul>
      </section>
    </article>
  );
}
