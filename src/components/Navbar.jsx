const navItems = [
  { label: 'About', page: 'about' },
  { label: 'Resume', page: 'resume' },
  { label: 'Projects', page: 'projects' },
  { label: 'Highlights', page: 'highlights' },
  { label: 'Contact', page: 'contact' },
];

export default function Navbar({ activePage, onNavigate }) {
  return (
    <nav className="navbar">
      <ul className="navbar-list">
        {navItems.map(({ label, page }) => (
          <li key={page} className="navbar-item">
            <button
              type="button"
              className={`navbar-link${activePage === page ? ' active' : ''}`}
              onClick={() => onNavigate(page)}
            >
              {label}
            </button>
          </li>
        ))}
      </ul>
    </nav>
  );
}
