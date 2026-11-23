// Inlines the question bank into the page. Run: node deep_dive/quiz/build.js
const fs = require('fs'), path = require('path');
const dir = __dirname;
const q = fs.readFileSync(path.join(dir, 'questions.js'), 'utf8').replace(/if \(typeof module[^\n]*\n/, '');
const html = fs.readFileSync(path.join(dir, 'page.src.html'), 'utf8').replace('/*QUESTIONS*/', () => q);
fs.writeFileSync(path.join(dir, 'systolic-quiz.html'), html);
console.log('wrote systolic-quiz.html', html.length, 'bytes');
