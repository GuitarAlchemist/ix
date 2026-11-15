package workshop

import (
	"errors"
)

// A device that administers the others from the app: the PC the owner linked from the owner page, or a device the
// owner granted from there. It lists the devices, invites one, approves, refuses or revokes it. It cannot revoke
// itself, so the owner page stays the only way to undo the owner's own access. Its authority is its device token,
// held by its browser: weaker than the owner page's local key, as accepted by the owner on 2026-11-14.

var ErrNotAdmin = errors.New("not_admin")

// DeviceView is a device as an administering device sees it.
type DeviceView struct {
	Device
	Self bool `json:"self"`
}

func (s *Store) isAdmin(id string) (bool, error) {
	var n int
	err := s.db.QueryRow(`SELECT count(*) FROM workshop_admins WHERE device_id=?`, id).Scan(&n)
	return n == 1, err
}

// Me is the device behind token, with whether it administers.
func (s *Store) Me(token string) (Device, error) {
	s.mu.Lock()
	defer s.mu.Unlock()
	d, err := authenticate(s.db, token)
	if err != nil {
		return d, err
	}
	if d.Admin, err = s.isAdmin(d.ID); err != nil {
		return d, err
	}
	d.Terminal, err = s.hasTerminal(d.ID)
	return d, err
}

// admin authenticates token and requires an administering device. The caller holds s.mu.
func (s *Store) admin(token string) (Device, error) {
	d, err := authenticate(s.db, token)
	if err != nil {
		return d, err
	}
	admin, err := s.isAdmin(d.ID)
	if err != nil {
		return d, err
	}
	if !admin {
		return d, ErrNotAdmin
	}
	return d, nil
}

func (s *Store) AdminDevices(token string) ([]DeviceView, error) {
	s.mu.Lock()
	defer s.mu.Unlock()
	me, err := s.admin(token)
	if err != nil {
		return nil, err
	}
	devices, err := listDevices(s.db)
	if err != nil {
		return nil, err
	}
	views := make([]DeviceView, 0, len(devices))
	for _, d := range devices {
		views = append(views, DeviceView{d, d.ID == me.ID})
	}
	return views, nil
}

// AdminInvite creates an ordinary invite: the device that claims it waits for approval.
func (s *Store) AdminInvite(token, origin string) (Invite, error) {
	s.mu.Lock()
	_, err := s.admin(token)
	s.mu.Unlock()
	if err != nil {
		return Invite{}, err
	}
	return s.NewInvite(origin)
}

func (s *Store) AdminDecide(token, id, status string) error {
	s.mu.Lock()
	me, err := s.admin(token)
	s.mu.Unlock()
	if err != nil {
		return err
	}
	if id == me.ID {
		return ErrInvalid
	}
	return s.Decide(id, status)
}

// GrantAdmin is owner-only (owner page): an approved device starts administering the others.
func (s *Store) GrantAdmin(id string) error {
	s.mu.Lock()
	defer s.mu.Unlock()
	var status string
	if err := s.db.QueryRow(`SELECT status FROM workshop_devices WHERE id=?`, id).Scan(&status); err != nil || status != "approved" {
		return ErrInvalid
	}
	_, err := s.db.Exec(`INSERT OR IGNORE INTO workshop_admins VALUES(?)`, id)
	return err
}
