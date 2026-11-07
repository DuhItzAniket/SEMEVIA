# Firebase Hosting front for the Cloud Run service

Gives the app the short address https://semevia.web.app (and https://semevia.firebaseapp.com). Firebase Hosting
forwards every request to the Cloud Run service `semevia` in `asia-south1` (see `docs/hosting_lock.md`); nothing is
served from `public/`, which is intentionally empty. Firebase Hosting has a 60-second request limit.

Deploy (team laptop, after `firebase login`):

```
cd deploy/firebase
C:\semevia\firebase-cli\node_modules\.bin\firebase.cmd deploy --only hosting --project semevia
```
