import { useState } from 'react';
import Sidebar from './components/Sidebar';
import Navbar from './components/Navbar';
import About from './components/About';
import Resume from './components/Resume';
import Projects from './components/Projects';
import Highlights from './components/Highlights';
import Contact from './components/Contact';
import ProjectModal from './components/ProjectModal';

const pages = [
  { key: 'about', Component: About },
  { key: 'resume', Component: Resume },
  { key: 'projects', Component: Projects },
  { key: 'highlights', Component: Highlights },
  { key: 'contact', Component: Contact },
];

export default function App() {
  const [activePage, setActivePage] = useState('about');
  const [selectedProject, setSelectedProject] = useState(null);
  const [modalOpen, setModalOpen] = useState(false);

  const handleNavigate = (page) => {
    setActivePage(page);
    window.scrollTo(0, 0);
  };

  const handleProjectClick = (project) => {
    setSelectedProject(project);
    setModalOpen(true);
  };

  const handleCloseModal = () => {
    setModalOpen(false);
  };

  return (
    <>
      <main>
        <Sidebar />

        <div className="main-content">
          <Navbar activePage={activePage} onNavigate={handleNavigate} />

          {pages.map(({ key, Component }) => (
            <Component
              key={key}
              isActive={activePage === key}
              {...(key === 'projects' ? { onProjectClick: handleProjectClick } : {})}
            />
          ))}
        </div>
      </main>

      <ProjectModal
        project={selectedProject}
        isOpen={modalOpen}
        onClose={handleCloseModal}
      />
    </>
  );
}
