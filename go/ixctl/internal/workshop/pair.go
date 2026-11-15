package workshop

import (
	"crypto/subtle"
	"database/sql"
	"fmt"
	"net/url"
	"strings"
	"time"
)

type Device struct {
	ID       string `json:"id"`
	Name     string `json:"name"`
	Status   string `json:"status"`
	Admin    bool   `json:"admin,omitempty"`    // administers the other devices from the app
	Terminal bool   `json:"terminal,omitempty"` // runs commands on the Desktop's PC (terminal.go)
}
type Invite struct {
	Code    string    `json:"-"`
	URL     string    `json:"url"`
	Expires time.Time `json:"expires"`
	QRSVG   string    `json:"qr_svg"`
}

func validOrigin(origin string) bool {
	u, err := url.Parse(origin)
	return err == nil && u.Scheme == "https" && u.Host != "" && u.User == nil && u.RawQuery == "" && u.Fragment == "" && u.Path == "" && u.Opaque == "" && u.String() == origin
}

func ValidateOrigin(origin string) error {
	if !validOrigin(origin) {
		return ErrInvalid
	}
	return nil
}

// NewInvite, Devices and Decide are owner-only Go calls for the existing Desktop host's trusted UI.
// There is deliberately no owner/admin endpoint in Handler. Never invoke these from a tablet request.
func (s *Store) NewInvite(origin string) (Invite, error) { return s.newInvite(origin, false) }

// NewLinkInvite is the owner linking the IxForge of the PC that hosts this Desktop. The owner's own action is the
// approval: the device that claims this invite is approved at once. Same secret, lifetime, single use and guess limit
// as any invite; its URL tells the client to claim it without asking for a code.
func (s *Store) NewLinkInvite(origin string) (Invite, error) { return s.newInvite(origin, true) }

func (s *Store) newInvite(origin string, linked bool) (Invite, error) {
	s.mu.Lock()
	defer s.mu.Unlock()
	var invite Invite
	if !validOrigin(origin) {
		return invite, ErrInvalid
	}
	code, err := randomID()
	if err != nil {
		return invite, err
	}
	code = code[:24]
	invite = Invite{Code: code, URL: origin + "/workshop/#pair=" + code, Expires: s.now().Add(2 * time.Minute)}
	rows, err := qrEncode(invite.URL)
	if err != nil {
		return Invite{}, err
	}
	var b strings.Builder
	n := len(rows) + 8
	fmt.Fprintf(&b, `<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 %d %d"><rect width="100%%" height="100%%" fill="white"/><g fill="black">`, n, n)
	for y, row := range rows {
		for x, v := range row {
			if v == '1' {
				fmt.Fprintf(&b, `<rect x="%d" y="%d" width="1" height="1"/>`, x+4, y+4)
			}
		}
	}
	b.WriteString("</g></svg>")
	invite.QRSVG = b.String()
	if linked {
		invite.URL += "&link=1"
	}
	s.inviteLinked = linked
	s.inviteHash = digest(code)
	s.inviteExpires = invite.Expires
	s.inviteErrors = 0
	return invite, nil
}
func (s *Store) Claim(code, name string) (Device, string, error) {
	s.mu.Lock()
	defer s.mu.Unlock()
	var d Device
	if !titleOK(name) || len(code) > 80 {
		return d, "", ErrInvalid
	}
	if s.inviteHash == "" || !s.now().Before(s.inviteExpires) {
		return d, "", ErrInvite
	}
	h := digest(code)
	if subtle.ConstantTimeCompare([]byte(h), []byte(s.inviteHash)) != 1 {
		s.inviteErrors++
		if s.inviteErrors >= 5 {
			s.inviteHash = ""
		}
		return d, "", ErrInvite
	}
	id, err := randomID()
	if err != nil {
		return d, "", err
	}
	token, err := randomID()
	if err != nil {
		return d, "", err
	}
	d = Device{ID: id, Name: name, Status: "pending"}
	if s.inviteLinked {
		d.Status = "approved"
	}
	if _, err = s.db.Exec(`INSERT INTO workshop_devices VALUES(?,?,?,?)`, d.ID, d.Name, digest(token), d.Status); err != nil {
		return Device{}, "", err
	}
	// The PC the owner links from the owner page also administers the other devices from the app.
	if s.inviteLinked {
		if _, err = s.db.Exec(`INSERT INTO workshop_admins VALUES(?)`, d.ID); err != nil {
			return Device{}, "", err
		}
		d.Admin = true
		// and has the terminal, which the owner allowed for the PC (terminal.go).
		if _, err = s.db.Exec(`INSERT INTO workshop_terminal VALUES(?)`, d.ID); err != nil {
			return Device{}, "", err
		}
		d.Terminal = true
	}
	s.inviteHash, s.inviteLinked = "", false
	return d, token, nil
}
func (s *Store) Devices() ([]Device, error) {
	s.mu.Lock()
	defer s.mu.Unlock()
	return listDevices(s.db)
}

func listDevices(db *sql.DB) ([]Device, error) {
	rows, err := db.Query(`SELECT d.id,d.name,d.status,a.device_id IS NOT NULL,x.device_id IS NOT NULL FROM workshop_devices d LEFT JOIN workshop_admins a ON a.device_id=d.id LEFT JOIN workshop_terminal x ON x.device_id=d.id ORDER BY d.rowid`)
	if err != nil {
		return nil, err
	}
	defer rows.Close()
	devices := []Device{}
	for rows.Next() {
		var d Device
		if err = rows.Scan(&d.ID, &d.Name, &d.Status, &d.Admin, &d.Terminal); err != nil {
			return nil, err
		}
		devices = append(devices, d)
	}
	return devices, rows.Err()
}
func (s *Store) Decide(id, status string) error {
	s.mu.Lock()
	defer s.mu.Unlock()
	var query string
	switch status {
	case "approved", "refused":
		query = `UPDATE workshop_devices SET status=? WHERE id=? AND status='pending'`
	case "revoked":
		query = `UPDATE workshop_devices SET status=? WHERE id=? AND status IN ('pending','approved')`
	default:
		return ErrInvalid
	}
	r, err := s.db.Exec(query, status, id)
	if err != nil {
		return err
	}
	n, err := r.RowsAffected()
	if err != nil {
		return err
	}
	if n != 1 {
		return ErrInvalid
	}
	return nil
}
