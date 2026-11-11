export const PRESETS = [
  { id:'china',         name:'China',           w:33, h:48, dpi:600, headW:[15,22], headH:[28,33], crown:[3,5], chin:7 },
  { id:'us',            name:'United States',   w:51, h:51, dpi:600, headW:[18,25], headH:[25,35], crown:[3,5], chin:7 },
  { id:'uk',            name:'United Kingdom',  w:35, h:45, dpi:600, headW:[16,20], headH:[29,34], crown:[2,5], chin:5 },
  { id:'canada',        name:'Canada',          w:50, h:70, dpi:600, headW:[19,24], headH:[31,36], crown:[3,5], chin:5 },
  { id:'australia',     name:'Australia',       w:35, h:45, dpi:600, headW:[20,24], headH:[32,36], crown:[3,5], chin:5 },
  { id:'france',        name:'France',          w:35, h:45, dpi:600, headW:[20,24], headH:[32,36], crown:[2,5], chin:5 },
  { id:'germany',       name:'Germany',         w:35, h:45, dpi:600, headW:[16,22], headH:[25,35], crown:[2,5], chin:5 },
  { id:'japan',         name:'Japan',           w:35, h:45, dpi:600, headW:[18,22], headH:[30,35], crown:[2,4], chin:5 },
  { id:'india',         name:'India',           w:35, h:45, dpi:600, headW:[16,22], headH:[25,35], crown:[3,5], chin:5 },
  { id:'brazil',        name:'Brazil',          w:35, h:45, dpi:600, headW:[16,22], headH:[25,35], crown:[2,5], chin:5 },
  { id:'eu',            name:'EU / Schengen',   w:35, h:45, dpi:600, headW:[16,20], headH:[29,34], crown:[2,5], chin:5 },
  { id:'korea',         name:'South Korea',     w:35, h:45, dpi:600, headW:[16,20], headH:[29,34], crown:[2,5], chin:5 },
  { id:'russia',        name:'Russia',          w:33, h:48, dpi:600, headW:[15,22], headH:[28,33], crown:[3,5], chin:7 },
  { id:'thailand',      name:'Thailand',        w:40, h:60, dpi:600, headW:[16,20], headH:[28,33], crown:[3,5], chin:5 },
  { id:'vietnam',       name:'Vietnam',         w:40, h:60, dpi:600, headW:[16,20], headH:[28,33], crown:[3,5], chin:5 },
  { id:'indonesia',     name:'Indonesia',       w:40, h:60, dpi:600, headW:[16,20], headH:[28,33], crown:[3,5], chin:5 },
  { id:'philippines',   name:'Philippines',     w:35, h:45, dpi:600, headW:[16,20], headH:[29,34], crown:[2,5], chin:5 },
  { id:'mexico',        name:'Mexico',          w:35, h:45, dpi:600, headW:[16,20], headH:[29,34], crown:[2,5], chin:5 },
  { id:'argentina',     name:'Argentina',       w:35, h:45, dpi:600, headW:[16,20], headH:[29,34], crown:[2,5], chin:5 },
  { id:'turkey',        name:'Turkey',          w:35, h:45, dpi:600, headW:[16,20], headH:[29,34], crown:[2,5], chin:5 },
  { id:'saudi',         name:'Saudi Arabia',    w:35, h:45, dpi:600, headW:[16,20], headH:[29,34], crown:[2,5], chin:5 },
  { id:'uae',           name:'UAE',             w:35, h:45, dpi:600, headW:[16,20], headH:[29,34], crown:[2,5], chin:5 },
  { id:'nigeria',       name:'Nigeria',         w:35, h:45, dpi:600, headW:[16,20], headH:[29,34], crown:[2,5], chin:5 },
  { id:'egypt',         name:'Egypt',           w:35, h:45, dpi:600, headW:[16,20], headH:[29,34], crown:[2,5], chin:5 },
];

export function getPreset(id) {
  return PRESETS.find(p => p.id === id) || PRESETS[0];
}
