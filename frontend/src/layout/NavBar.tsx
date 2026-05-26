import { NavLink } from 'react-router-dom';
import type { CSSProperties } from 'react';

const navStyle = (active: boolean): CSSProperties => ({
  padding: '8px 16px',
  color: active ? '#fff' : '#aaa',
  background: active ? '#222' : 'transparent',
  borderRadius: 4,
  textDecoration: 'none',
  fontSize: 14,
});

export function NavBar() {
  return (
    <nav style={{
      display: 'flex',
      gap: 8,
      padding: '8px 16px',
      background: '#0c0c0c',
      borderBottom: '1px solid #222',
    }}>
      <NavLink to="/" end style={({ isActive }) => navStyle(isActive)}>Live</NavLink>
      <NavLink to="/people" style={({ isActive }) => navStyle(isActive)}>People</NavLink>
      <NavLink to="/review" style={({ isActive }) => navStyle(isActive)}>Review</NavLink>
    </nav>
  );
}
