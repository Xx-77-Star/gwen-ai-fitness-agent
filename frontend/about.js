import { AvatarController } from "./avatar/AvatarController.js";
import { HERO_AVATAR_CONFIG } from "./avatar/avatar-config.js";

// About page: a quiet, passive Gwen model - no rotate/zoom, no chat wiring.
const mount = document.querySelector("#about-avatar");
if (mount) {
  new AvatarController(mount, HERO_AVATAR_CONFIG, {
    interactive: false,
    viewerControls: false,
    readyHint: "",
    idleHint: "",
  });
}
