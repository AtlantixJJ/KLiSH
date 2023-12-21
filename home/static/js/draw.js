'use strict';

function readTextFile(file, callback) {
  var rawFile = new XMLHttpRequest();
  rawFile.overrideMimeType("application/json");
  rawFile.open("GET", file, true);
  rawFile.onreadystatechange = function () {
    if (rawFile.readyState === 4 && rawFile.status == "200") {
      callback(rawFile.responseText);
    }
  }
  rawFile.send(null);
}

var graph = null, // canvas manager
  labelgraph = null; // label canvas manager
var currentModel = 0; // current model index
var loading = false; // the image is loading (waiting response)
var image = null, // image data
  label = null; // label image data
//latent = null, // latent data
//noise = null; // noise data
var imheight = 0, imwidth = 0;
var config = null; // config file
var use_args = false; // [deprecated]
var spinner = new Spinner({ color: '#999' });
var painter_colors = [
  'black',
  'rgb(208, 2, 27)',
  'rgb(245, 166, 35)',
  'rgb(248, 231, 28)',
  'rgb(139, 87, 42)',
  'rgb(126, 211, 33)',
  'white',
  'rgb(226, 238, 244)',
  'rgb(226, 178, 213)',
  'rgb(189, 16, 224)',
  'rgb(74, 144, 226)',
  'rgb(80, 227, 194)'];

/*
var FACE_CATEGORY = ['background', 'skin', 'nose', 'eye glasses', 'eye', 'brow', 'ear', 'mouth', 'upper lip', 'lower lip', 'hair', 'hat', 'ear rings', 'neck', 'cloth'];
var STYLEGAN2_BEDROOM_CATEGORY = ['wall','floor','ceiling','bed','windowpane','table','curtain','painting','lamp','cushion','pillow','flower','light','chandelier','fan','clock'];

var FACE_CATEGORY_COLORS = [
  'rgb(0, 0, 0)', 'rgb(255, 255, 0)', 'rgb(28, 230, 255)',
  'rgb(255, 52, 255)', 'rgb(255, 74, 70)', 'rgb(0, 137, 65)',
  'rgb(0, 111, 166)', 'rgb(163, 0, 89)', 'rgb(255, 219, 229)',
  'rgb(122, 73, 0)', 'rgb(0, 0, 166)', 'rgb(99, 255, 172)',
  'rgb(183, 151, 98)', 'rgb(0, 77, 67)', 'rgb(143, 176, 255)'];
var STYLEGAN2_BEDROOM_COLORS = [
  'rgb(255, 255, 0)', 'rgb(28, 230, 255)',
  'rgb(255, 52, 255)', 'rgb(255, 74, 70)', 'rgb(0, 137, 65)',
  'rgb(0, 111, 166)', 'rgb(163, 0, 89)', 'rgb(255, 219, 229)',
  'rgb(122, 73, 0)', 'rgb(0, 0, 166)', 'rgb(99, 255, 172)',
  'rgb(183, 151, 98)', 'rgb(0, 77, 67)', 'rgb(143, 176, 255)'];
*/

var HIGH_CONSTRAST = [
  'rgb(0, 0, 0)', 'rgb(255, 255, 0)', 'rgb(28, 230, 255)', 'rgb(255, 52, 255)', 'rgb(255, 74, 70)', 'rgb(0, 137, 65)', 'rgb(0, 111, 166)', 'rgb(163, 0, 89)', 'rgb(255, 219, 229)', 'rgb(122, 73, 0)', 'rgb(0, 0, 166)', 'rgb(99, 255, 172)', 'rgb(183, 151, 98)', 'rgb(0, 77, 67)', 'rgb(143, 176, 255)', 'rgb(153, 125, 135)', 'rgb(90, 0, 7)', 'rgb(128, 150, 147)', 'rgb(254, 255, 230)', 'rgb(27, 68, 0)', 'rgb(79, 198, 1)', 'rgb(59, 93, 255)', 'rgb(74, 59, 83)', 'rgb(255, 47, 128)', 'rgb(97, 97, 90)', 'rgb(186, 9, 0)', 'rgb(107, 121, 0)', 'rgb(0, 194, 160)', 'rgb(255, 170, 146)', 'rgb(255, 144, 201)', 'rgb(185, 3, 170)', 'rgb(209, 97, 0)', 'rgb(221, 239, 255)', 'rgb(0, 0, 53)', 'rgb(123, 79, 75)', 'rgb(161, 194, 153)', 'rgb(48, 0, 24)', 'rgb(10, 166, 216)', 'rgb(1, 51, 73)', 'rgb(0, 132, 111)', 'rgb(55, 33, 1)', 'rgb(255, 181, 0)', 'rgb(194, 255, 237)', 'rgb(160, 121, 191)', 'rgb(204, 7, 68)', 'rgb(192, 185, 178)', 'rgb(194, 255, 153)', 'rgb(0, 30, 9)', 'rgb(0, 72, 156)', 'rgb(111, 0, 98)', 'rgb(12, 189, 102)', 'rgb(238, 195, 255)', 'rgb(69, 109, 117)', 'rgb(183, 123, 104)', 'rgb(122, 135, 161)', 'rgb(120, 141, 102)', 'rgb(136, 85, 120)', 'rgb(250, 208, 159)', 'rgb(255, 138, 154)', 'rgb(209, 87, 160)', 'rgb(190, 196, 89)', 'rgb(69, 102, 72)', 'rgb(0, 134, 237)', 'rgb(136, 111, 76)', 'rgb(52, 54, 45)', 'rgb(180, 168, 189)', 'rgb(0, 166, 170)', 'rgb(69, 44, 44)', 'rgb(99, 99, 117)', 'rgb(163, 200, 201)', 'rgb(255, 145, 63)', 'rgb(147, 138, 129)', 'rgb(87, 83, 41)', 'rgb(0, 254, 207)', 'rgb(176, 91, 111)', 'rgb(140, 208, 255)', 'rgb(59, 151, 0)', 'rgb(4, 247, 87)', 'rgb(200, 161, 161)', 'rgb(30, 110, 0)', 'rgb(121, 0, 215)', 'rgb(167, 117, 0)', 'rgb(99, 103, 169)', 'rgb(160, 88, 55)', 'rgb(107, 0, 44)', 'rgb(119, 38, 0)', 'rgb(215, 144, 255)', 'rgb(155, 151, 0)', 'rgb(84, 158, 121)', 'rgb(255, 246, 159)', 'rgb(32, 22, 37)', 'rgb(114, 65, 143)', 'rgb(188, 35, 255)', 'rgb(153, 173, 192)', 'rgb(58, 36, 101)', 'rgb(146, 35, 41)', 'rgb(91, 69, 52)', 'rgb(253, 232, 220)', 'rgb(64, 78, 85)', 'rgb(0, 137, 163)', 'rgb(203, 126, 152)', 'rgb(164, 232, 4)', 'rgb(50, 78, 114)', 'rgb(106, 58, 76)', 'rgb(131, 171, 88)', 'rgb(0, 28, 30)', 'rgb(209, 247, 206)', 'rgb(0, 75, 40)', 'rgb(200, 208, 246)', 'rgb(163, 164, 137)', 'rgb(128, 108, 102)', 'rgb(34, 40, 0)', 'rgb(191, 86, 80)', 'rgb(232, 48, 0)', 'rgb(102, 121, 109)', 'rgb(218, 0, 124)', 'rgb(255, 26, 89)', 'rgb(138, 219, 180)', 'rgb(30, 2, 0)', 'rgb(91, 78, 81)', 'rgb(200, 149, 197)', 'rgb(50, 0, 51)', 'rgb(255, 104, 50)', 'rgb(102, 225, 211)', 'rgb(207, 205, 172)', 'rgb(208, 172, 148)', 'rgb(126, 211, 121)', 'rgb(1, 44, 88)', 'rgb(122, 123, 255)', 'rgb(214, 142, 1)', 'rgb(53, 51, 57)', 'rgb(120, 175, 161)', 'rgb(254, 178, 198)', 'rgb(117, 121, 124)', 'rgb(131, 115, 147)', 'rgb(148, 58, 77)', 'rgb(181, 244, 255)', 'rgb(210, 220, 213)', 'rgb(149, 86, 189)', 'rgb(106, 113, 74)', 'rgb(0, 19, 37)', 'rgb(2, 82, 95)', 'rgb(10, 163, 247)', 'rgb(233, 129, 118)', 'rgb(219, 213, 221)', 'rgb(94, 188, 209)', 'rgb(61, 79, 68)', 'rgb(126, 100, 5)', 'rgb(2, 104, 78)', 'rgb(150, 43, 117)', 'rgb(141, 133, 70)', 'rgb(150, 149, 197)', 'rgb(231, 115, 206)', 'rgb(216, 106, 120)', 'rgb(62, 137, 190)', 'rgb(202, 131, 78)', 'rgb(81, 138, 135)', 'rgb(91, 17, 60)', 'rgb(85, 129, 59)', 'rgb(231, 4, 196)', 'rgb(0, 0, 95)', 'rgb(169, 115, 153)', 'rgb(75, 129, 96)', 'rgb(89, 115, 138)', 'rgb(255, 93, 167)', 'rgb(247, 201, 191)', 'rgb(100, 49, 39)', 'rgb(81, 58, 1)', 'rgb(107, 148, 170)', 'rgb(81, 160, 88)', 'rgb(164, 91, 2)', 'rgb(29, 23, 2)', 'rgb(226, 0, 39)', 'rgb(231, 171, 99)', 'rgb(76, 96, 1)', 'rgb(156, 105, 102)', 'rgb(100, 84, 123)', 'rgb(151, 151, 158)', 'rgb(0, 106, 102)', 'rgb(57, 20, 6)', 'rgb(244, 215, 73)', 'rgb(0, 69, 210)', 'rgb(0, 108, 49)', 'rgb(221, 182, 208)', 'rgb(124, 101, 113)', 'rgb(159, 178, 164)', 'rgb(0, 216, 145)', 'rgb(21, 160, 138)', 'rgb(188, 101, 233)', 'rgb(255, 255, 254)', 'rgb(198, 220, 153)', 'rgb(32, 59, 60)', 'rgb(103, 17, 144)', 'rgb(107, 58, 100)', 'rgb(245, 225, 255)', 'rgb(255, 160, 242)', 'rgb(204, 170, 53)', 'rgb(55, 69, 39)', 'rgb(139, 180, 0)', 'rgb(121, 120, 104)', 'rgb(198, 0, 90)', 'rgb(59, 0, 10)', 'rgb(200, 98, 64)', 'rgb(41, 96, 124)', 'rgb(64, 35, 52)', 'rgb(125, 90, 68)', 'rgb(204, 184, 124)', 'rgb(184, 129, 131)', 'rgb(170, 81, 153)', 'rgb(181, 214, 195)', 'rgb(163, 132, 105)', 'rgb(159, 148, 240)', 'rgb(167, 69, 113)', 'rgb(184, 148, 166)', 'rgb(113, 187, 140)', 'rgb(0, 180, 51)', 'rgb(120, 158, 201)', 'rgb(109, 128, 186)', 'rgb(149, 63, 0)', 'rgb(94, 255, 3)', 'rgb(228, 255, 252)', 'rgb(27, 225, 119)', 'rgb(188, 177, 229)', 'rgb(118, 145, 47)', 'rgb(0, 49, 9)', 'rgb(0, 96, 205)', 'rgb(210, 0, 150)', 'rgb(137, 85, 99)', 'rgb(41, 32, 29)', 'rgb(91, 50, 19)', 'rgb(167, 111, 66)', 'rgb(137, 65, 46)', 'rgb(26, 58, 42)', 'rgb(73, 75, 90)', 'rgb(168, 140, 133)', 'rgb(244, 171, 170)', 'rgb(163, 243, 171)', 'rgb(0, 198, 200)', 'rgb(234, 139, 102)', 'rgb(149, 138, 159)', 'rgb(189, 201, 210)', 'rgb(159, 160, 100)', 'rgb(190, 71, 0)', 'rgb(101, 129, 136)', 'rgb(131, 164, 133)', 'rgb(69, 60, 35)', 'rgb(71, 103, 93)', 'rgb(58, 63, 0)', 'rgb(6, 18, 3)', 'rgb(223, 251, 113)', 'rgb(134, 142, 126)', 'rgb(152, 208, 88)', 'rgb(108, 143, 125)', 'rgb(215, 191, 194)', 'rgb(60, 62, 110)', 'rgb(216, 61, 102)', 'rgb(47, 93, 155)', 'rgb(108, 94, 70)', 'rgb(210, 91, 136)', 'rgb(91, 101, 108)', 'rgb(0, 181, 127)', 'rgb(84, 92, 70)', 'rgb(134, 96, 151)', 'rgb(54, 93, 37)', 'rgb(37, 47, 153)', 'rgb(0, 204, 255)', 'rgb(103, 78, 96)', 'rgb(252, 0, 156)', 'rgb(146, 137, 107)', 'rgb(30, 35, 36)', 'rgb(222, 201, 178)', 'rgb(157, 73, 72)', 'rgb(133, 171, 180)', 'rgb(52, 33, 66)', 'rgb(208, 150, 133)', 'rgb(164, 172, 172)', 'rgb(0, 255, 255)', 'rgb(174, 156, 134)', 'rgb(116, 42, 51)', 'rgb(14, 114, 197)', 'rgb(175, 216, 236)', 'rgb(192, 100, 185)', 'rgb(145, 2, 140)', 'rgb(254, 237, 191)', 'rgb(255, 183, 137)', 'rgb(156, 184, 228)', 'rgb(175, 255, 209)', 'rgb(42, 54, 76)', 'rgb(79, 74, 67)', 'rgb(100, 112, 149)', 'rgb(52, 187, 255)', 'rgb(128, 119, 129)', 'rgb(146, 0, 3)', 'rgb(179, 165, 167)', 'rgb(1, 134, 21)', 'rgb(241, 255, 200)', 'rgb(151, 111, 92)', 'rgb(255, 59, 193)', 'rgb(255, 95, 107)', 'rgb(7, 125, 132)', 'rgb(245, 109, 147)', 'rgb(87, 113, 218)', 'rgb(78, 30, 42)', 'rgb(131, 0, 85)', 'rgb(2, 211, 70)', 'rgb(190, 69, 45)', 'rgb(0, 144, 94)', 'rgb(190, 0, 40)', 'rgb(110, 150, 227)', 'rgb(0, 118, 153)', 'rgb(254, 201, 109)', 'rgb(156, 106, 125)', 'rgb(63, 161, 184)', 'rgb(137, 61, 227)', 'rgb(121, 180, 214)', 'rgb(127, 212, 217)', 'rgb(103, 81, 187)', 'rgb(178, 141, 45)', 'rgb(226, 122, 5)', 'rgb(221, 156, 184)', 'rgb(170, 188, 122)', 'rgb(152, 0, 52)', 'rgb(86, 26, 2)', 'rgb(143, 127, 0)', 'rgb(99, 80, 0)', 'rgb(205, 125, 174)', 'rgb(138, 94, 45)', 'rgb(255, 179, 225)', 'rgb(107, 100, 102)', 'rgb(198, 211, 0)', 'rgb(1, 0, 226)', 'rgb(136, 236, 105)', 'rgb(143, 204, 190)', 'rgb(33, 0, 28)', 'rgb(81, 31, 77)', 'rgb(227, 246, 227)', 'rgb(255, 142, 177)', 'rgb(107, 79, 41)', 'rgb(163, 127, 70)', 'rgb(106, 89, 80)', 'rgb(31, 42, 26)', 'rgb(4, 120, 77)', 'rgb(16, 24, 53)', 'rgb(230, 224, 208)', 'rgb(255, 116, 254)', 'rgb(0, 164, 95)', 'rgb(143, 93, 248)', 'rgb(75, 0, 89)', 'rgb(65, 47, 35)', 'rgb(216, 147, 158)', 'rgb(219, 157, 114)', 'rgb(96, 65, 67)', 'rgb(181, 186, 206)', 'rgb(152, 158, 183)', 'rgb(210, 196, 219)', 'rgb(165, 135, 175)', 'rgb(119, 215, 150)', 'rgb(127, 140, 148)', 'rgb(255, 155, 3)', 'rgb(85, 81, 150)', 'rgb(49, 221, 174)', 'rgb(116, 182, 113)', 'rgb(128, 38, 71)', 'rgb(42, 55, 63)', 'rgb(1, 74, 104)', 'rgb(105, 102, 40)', 'rgb(76, 123, 109)', 'rgb(0, 44, 39)', 'rgb(122, 69, 34)', 'rgb(59, 88, 89)', 'rgb(229, 211, 129)', 'rgb(255, 243, 255)', 'rgb(103, 159, 160)', 'rgb(38, 19, 0)', 'rgb(44, 87, 66)', 'rgb(145, 49, 175)', 'rgb(175, 93, 136)', 'rgb(199, 112, 106)', 'rgb(97, 171, 31)', 'rgb(140, 242, 212)', 'rgb(197, 217, 184)', 'rgb(159, 255, 251)', 'rgb(191, 69, 204)', 'rgb(73, 57, 65)', 'rgb(134, 59, 96)', 'rgb(185, 0, 118)', 'rgb(0, 49, 119)', 'rgb(197, 130, 210)', 'rgb(193, 179, 148)', 'rgb(96, 43, 112)', 'rgb(136, 120, 104)', 'rgb(186, 191, 176)', 'rgb(3, 0, 18)', 'rgb(209, 172, 254)', 'rgb(127, 222, 254)', 'rgb(75, 92, 113)', 'rgb(163, 160, 151)', 'rgb(230, 109, 83)', 'rgb(99, 123, 93)', 'rgb(146, 190, 165)', 'rgb(0, 248, 179)', 'rgb(190, 221, 255)', 'rgb(61, 181, 167)', 'rgb(221, 50, 72)', 'rgb(182, 228, 222)', 'rgb(66, 119, 69)', 'rgb(89, 140, 90)', 'rgb(185, 76, 89)', 'rgb(129, 129, 213)', 'rgb(148, 136, 139)', 'rgb(254, 214, 189)', 'rgb(83, 109, 49)', 'rgb(110, 255, 146)', 'rgb(228, 232, 255)', 'rgb(32, 226, 0)', 'rgb(255, 208, 242)', 'rgb(76, 131, 161)', 'rgb(189, 115, 34)', 'rgb(145, 92, 78)', 'rgb(140, 71, 135)', 'rgb(2, 81, 23)', 'rgb(162, 170, 69)', 'rgb(45, 27, 33)', 'rgb(169, 221, 176)', 'rgb(255, 79, 120)', 'rgb(82, 133, 0)', 'rgb(0, 154, 46)', 'rgb(23, 252, 228)', 'rgb(113, 85, 90)', 'rgb(82, 93, 130)', 'rgb(0, 25, 90)', 'rgb(150, 120, 116)', 'rgb(85, 85, 88)', 'rgb(11, 33, 44)', 'rgb(30, 32, 43)', 'rgb(239, 191, 196)', 'rgb(111, 151, 85)', 'rgb(111, 117, 134)', 'rgb(80, 29, 29)', 'rgb(55, 45, 0)', 'rgb(116, 29, 22)', 'rgb(94, 179, 147)', 'rgb(181, 180, 0)', 'rgb(221, 74, 56)', 'rgb(54, 61, 255)', 'rgb(173, 101, 82)', 'rgb(102, 53, 175)', 'rgb(131, 107, 186)', 'rgb(152, 170, 127)', 'rgb(70, 72, 54)', 'rgb(50, 44, 62)', 'rgb(124, 185, 186)', 'rgb(91, 105, 101)', 'rgb(112, 125, 61)', 'rgb(122, 0, 29)', 'rgb(110, 70, 54)', 'rgb(68, 58, 56)', 'rgb(174, 129, 255)', 'rgb(72, 144, 121)', 'rgb(137, 115, 52)', 'rgb(0, 144, 135)', 'rgb(218, 113, 60)', 'rgb(54, 22, 24)', 'rgb(255, 111, 1)', 'rgb(0, 102, 121)', 'rgb(55, 14, 119)', 'rgb(75, 58, 131)', 'rgb(201, 226, 230)', 'rgb(196, 65, 112)', 'rgb(255, 69, 38)', 'rgb(115, 190, 84)', 'rgb(196, 223, 114)', 'rgb(173, 255, 96)', 'rgb(0, 68, 125)', 'rgb(220, 206, 201)', 'rgb(189, 148, 121)', 'rgb(101, 110, 91)', 'rgb(236, 82, 0)', 'rgb(255, 110, 194)', 'rgb(122, 97, 126)', 'rgb(221, 174, 162)', 'rgb(119, 131, 127)', 'rgb(165, 51, 39)', 'rgb(96, 142, 255)', 'rgb(181, 153, 215)', 'rgb(165, 1, 73)', 'rgb(78, 0, 37)', 'rgb(201, 177, 169)', 'rgb(3, 145, 154)', 'rgb(27, 42, 37)', 'rgb(229, 0, 241)', 'rgb(152, 46, 11)', 'rgb(182, 113, 128)', 'rgb(224, 88, 89)', 'rgb(0, 96, 57)', 'rgb(87, 143, 155)', 'rgb(48, 82, 48)', 'rgb(206, 147, 76)', 'rgb(179, 194, 190)', 'rgb(192, 186, 192)', 'rgb(181, 6, 211)', 'rgb(23, 12, 16)', 'rgb(76, 83, 79)', 'rgb(34, 68, 81)', 'rgb(62, 65, 65)', 'rgb(120, 114, 109)', 'rgb(182, 96, 43)', 'rgb(32, 4, 65)', 'rgb(221, 181, 136)', 'rgb(73, 114, 0)', 'rgb(197, 170, 182)', 'rgb(3, 60, 97)', 'rgb(113, 178, 245)', 'rgb(169, 224, 136)', 'rgb(73, 121, 176)', 'rgb(162, 195, 223)', 'rgb(120, 65, 73)', 'rgb(45, 43, 23)', 'rgb(62, 14, 47)', 'rgb(87, 52, 76)', 'rgb(0, 145, 190)', 'rgb(228, 81, 209)', 'rgb(75, 75, 106)', 'rgb(92, 1, 26)', 'rgb(124, 128, 96)', 'rgb(255, 148, 145)', 'rgb(76, 50, 93)', 'rgb(0, 92, 139)', 'rgb(229, 253, 164)', 'rgb(104, 209, 182)', 'rgb(3, 38, 65)', 'rgb(20, 0, 35)', 'rgb(134, 131, 169)', 'rgb(207, 255, 0)', 'rgb(167, 44, 62)', 'rgb(52, 71, 90)', 'rgb(177, 187, 154)', 'rgb(180, 160, 79)', 'rgb(141, 145, 142)', 'rgb(161, 104, 166)', 'rgb(129, 61, 58)', 'rgb(66, 82, 24)', 'rgb(218, 131, 134)', 'rgb(119, 97, 51)', 'rgb(86, 57, 48)', 'rgb(132, 152, 174)', 'rgb(144, 193, 211)', 'rgb(181, 102, 107)', 'rgb(155, 88, 94)', 'rgb(133, 100, 101)', 'rgb(173, 124, 144)', 'rgb(226, 188, 0)', 'rgb(227, 170, 224)', 'rgb(178, 194, 254)', 'rgb(253, 0, 57)', 'rgb(0, 155, 117)', 'rgb(255, 244, 109)', 'rgb(232, 126, 172)', 'rgb(223, 227, 230)', 'rgb(132, 133, 144)', 'rgb(170, 146, 151)', 'rgb(131, 161, 147)', 'rgb(87, 121, 119)', 'rgb(62, 113, 88)', 'rgb(198, 66, 137)', 'rgb(234, 0, 114)', 'rgb(196, 168, 203)', 'rgb(85, 200, 153)', 'rgb(231, 143, 207)', 'rgb(0, 69, 71)', 'rgb(246, 226, 227)', 'rgb(150, 103, 22)', 'rgb(55, 143, 219)', 'rgb(67, 94, 106)', 'rgb(218, 0, 4)', 'rgb(27, 0, 15)', 'rgb(91, 156, 143)', 'rgb(110, 43, 82)', 'rgb(1, 17, 21)', 'rgb(227, 232, 196)', 'rgb(174, 59, 133)', 'rgb(234, 28, 169)', 'rgb(255, 158, 107)', 'rgb(69, 125, 139)', 'rgb(146, 103, 139)', 'rgb(0, 205, 187)', 'rgb(156, 204, 4)', 'rgb(0, 46, 56)', 'rgb(150, 197, 127)', 'rgb(207, 246, 180)', 'rgb(73, 40, 24)', 'rgb(118, 110, 82)', 'rgb(32, 55, 14)', 'rgb(227, 209, 159)', 'rgb(46, 60, 48)', 'rgb(178, 234, 206)', 'rgb(243, 189, 164)', 'rgb(162, 78, 61)', 'rgb(151, 111, 217)', 'rgb(140, 159, 168)', 'rgb(124, 43, 115)', 'rgb(78, 95, 55)', 'rgb(93, 84, 98)', 'rgb(144, 149, 111)', 'rgb(106, 167, 118)', 'rgb(219, 203, 246)', 'rgb(218, 113, 255)', 'rgb(152, 124, 149)', 'rgb(82, 50, 60)', 'rgb(187, 60, 66)', 'rgb(88, 77, 57)', 'rgb(79, 193, 95)', 'rgb(162, 185, 193)', 'rgb(121, 219, 33)', 'rgb(29, 89, 88)', 'rgb(189, 116, 78)', 'rgb(22, 11, 0)', 'rgb(32, 34, 26)', 'rgb(107, 130, 149)', 'rgb(0, 224, 228)', 'rgb(16, 36, 1)', 'rgb(27, 120, 42)', 'rgb(218, 169, 181)', 'rgb(176, 65, 93)', 'rgb(133, 146, 83)', 'rgb(151, 160, 148)', 'rgb(6, 227, 196)', 'rgb(71, 104, 140)', 'rgb(124, 103, 85)', 'rgb(7, 92, 0)', 'rgb(117, 96, 213)', 'rgb(125, 159, 0)', 'rgb(195, 109, 150)', 'rgb(77, 145, 62)', 'rgb(95, 66, 118)', 'rgb(252, 228, 200)', 'rgb(48, 48, 82)', 'rgb(79, 56, 27)', 'rgb(229, 165, 50)', 'rgb(112, 102, 144)', 'rgb(170, 154, 146)', 'rgb(35, 115, 99)', 'rgb(115, 1, 62)', 'rgb(255, 144, 121)', 'rgb(167, 154, 116)', 'rgb(2, 155, 219)', 'rgb(255, 1, 105)', 'rgb(199, 210, 231)', 'rgb(202, 136, 105)', 'rgb(128, 255, 205)', 'rgb(187, 31, 105)', 'rgb(144, 176, 171)', 'rgb(125, 116, 169)', 'rgb(252, 199, 219)', 'rgb(153, 55, 91)', 'rgb(0, 171, 77)', 'rgb(171, 174, 209)', 'rgb(190, 157, 145)', 'rgb(230, 229, 167)', 'rgb(51, 44, 34)', 'rgb(221, 88, 123)', 'rgb(245, 255, 247)', 'rgb(93, 48, 51)', 'rgb(109, 56, 0)', 'rgb(255, 0, 32)', 'rgb(181, 123, 179)', 'rgb(215, 255, 230)', 'rgb(197, 53, 169)', 'rgb(38, 0, 9)', 'rgb(106, 135, 129)', 'rgb(168, 171, 180)', 'rgb(212, 82, 98)', 'rgb(121, 75, 97)', 'rgb(70, 33, 178)', 'rgb(141, 164, 219)', 'rgb(199, 200, 144)', 'rgb(111, 233, 173)', 'rgb(162, 67, 167)', 'rgb(178, 176, 129)', 'rgb(24, 27, 0)', 'rgb(40, 97, 84)', 'rgb(76, 164, 59)', 'rgb(106, 149, 115)', 'rgb(168, 68, 29)', 'rgb(92, 114, 123)', 'rgb(115, 134, 113)', 'rgb(208, 207, 203)', 'rgb(137, 123, 119)', 'rgb(31, 63, 34)', 'rgb(65, 69, 167)', 'rgb(218, 152, 148)', 'rgb(161, 117, 122)', 'rgb(99, 36, 60)', 'rgb(173, 170, 255)', 'rgb(0, 205, 226)', 'rgb(221, 188, 98)', 'rgb(105, 142, 177)', 'rgb(32, 132, 98)', 'rgb(0, 183, 224)', 'rgb(97, 74, 68)', 'rgb(155, 187, 87)', 'rgb(122, 92, 84)', 'rgb(133, 122, 80)', 'rgb(118, 107, 126)', 'rgb(1, 72, 51)', 'rgb(255, 131, 71)', 'rgb(122, 142, 186)', 'rgb(39, 71, 64)', 'rgb(148, 100, 68)', 'rgb(235, 216, 230)', 'rgb(100, 98, 65)', 'rgb(55, 57, 23)', 'rgb(106, 212, 80)', 'rgb(129, 129, 123)', 'rgb(212, 153, 227)', 'rgb(151, 148, 64)', 'rgb(1, 26, 18)', 'rgb(82, 101, 84)', 'rgb(181, 136, 92)', 'rgb(164, 153, 165)', 'rgb(3, 173, 137)', 'rgb(179, 0, 139)', 'rgb(227, 196, 181)', 'rgb(150, 83, 31)', 'rgb(134, 113, 117)', 'rgb(116, 86, 158)', 'rgb(97, 125, 159)', 'rgb(231, 4, 82)', 'rgb(6, 126, 175)', 'rgb(166, 151, 182)', 'rgb(183, 135, 168)', 'rgb(156, 255, 147)', 'rgb(49, 29, 25)', 'rgb(58, 148, 89)', 'rgb(110, 116, 110)', 'rgb(176, 197, 174)', 'rgb(132, 237, 247)', 'rgb(237, 52, 136)', 'rgb(117, 76, 120)', 'rgb(56, 70, 68)', 'rgb(199, 132, 123)', 'rgb(0, 182, 197)', 'rgb(127, 166, 112)', 'rgb(193, 175, 158)', 'rgb(42, 127, 255)', 'rgb(114, 165, 140)', 'rgb(255, 192, 127)', 'rgb(157, 235, 221)', 'rgb(217, 124, 142)', 'rgb(126, 124, 147)', 'rgb(98, 230, 116)', 'rgb(181, 99, 158)', 'rgb(255, 168, 97)', 'rgb(194, 165, 128)', 'rgb(141, 156, 131)', 'rgb(183, 5, 70)', 'rgb(55, 43, 46)', 'rgb(0, 152, 255)', 'rgb(152, 89, 117)', 'rgb(32, 32, 76)', 'rgb(255, 108, 96)', 'rgb(68, 80, 131)', 'rgb(133, 2, 170)', 'rgb(114, 54, 31)', 'rgb(150, 118, 163)', 'rgb(72, 68, 73)', 'rgb(206, 214, 194)', 'rgb(59, 22, 74)', 'rgb(204, 167, 99)', 'rgb(44, 127, 119)', 'rgb(2, 34, 123)', 'rgb(163, 126, 111)', 'rgb(205, 230, 220)', 'rgb(205, 255, 251)', 'rgb(190, 129, 26)', 'rgb(247, 113, 131)', 'rgb(237, 230, 226)', 'rgb(205, 198, 180)', 'rgb(255, 224, 158)', 'rgb(58, 114, 113)', 'rgb(255, 123, 89)', 'rgb(78, 78, 1)', 'rgb(74, 198, 132)', 'rgb(139, 200, 145)', 'rgb(188, 138, 150)', 'rgb(207, 99, 83)', 'rgb(220, 222, 92)', 'rgb(94, 170, 221)', 'rgb(246, 160, 173)', 'rgb(226, 105, 170)', 'rgb(163, 218, 228)', 'rgb(67, 110, 131)', 'rgb(0, 46, 23)', 'rgb(236, 251, 255)', 'rgb(161, 194, 182)', 'rgb(80, 0, 63)', 'rgb(113, 105, 91)', 'rgb(103, 196, 187)', 'rgb(83, 110, 255)', 'rgb(93, 90, 72)', 'rgb(137, 0, 57)', 'rgb(150, 147, 129)', 'rgb(55, 21, 33)', 'rgb(94, 70, 101)', 'rgb(170, 98, 195)', 'rgb(141, 111, 129)', 'rgb(44, 97, 53)', 'rgb(65, 6, 1)', 'rgb(86, 70, 32)', 'rgb(230, 144, 52)', 'rgb(109, 166, 189)', 'rgb(229, 142, 86)', 'rgb(227, 166, 139)', 'rgb(72, 177, 118)', 'rgb(210, 125, 103)', 'rgb(181, 178, 104)', 'rgb(127, 132, 39)', 'rgb(255, 132, 230)', 'rgb(67, 87, 64)', 'rgb(234, 228, 8)', 'rgb(244, 245, 255)', 'rgb(50, 88, 0)', 'rgb(75, 107, 165)', 'rgb(173, 206, 255)', 'rgb(155, 138, 204)', 'rgb(136, 81, 56)', 'rgb(88, 117, 193)', 'rgb(126, 115, 17)', 'rgb(254, 165, 202)', 'rgb(159, 139, 91)', 'rgb(165, 91, 84)', 'rgb(137, 0, 106)', 'rgb(175, 117, 111)', 'rgb(42, 32, 0)', 'rgb(116, 153, 161)', 'rgb(255, 181, 80)', 'rgb(0, 1, 30)', 'rgb(209, 81, 28)', 'rgb(104, 129, 81)', 'rgb(188, 144, 138)', 'rgb(120, 200, 235)', 'rgb(133, 2, 255)', 'rgb(72, 61, 48)', 'rgb(196, 34, 33)', 'rgb(94, 167, 255)', 'rgb(120, 87, 21)', 'rgb(12, 234, 145)', 'rgb(255, 250, 237)', 'rgb(179, 175, 157)', 'rgb(62, 61, 82)', 'rgb(90, 155, 194)', 'rgb(156, 47, 144)', 'rgb(141, 87, 0)', 'rgb(173, 215, 156)', 'rgb(0, 118, 139)', 'rgb(51, 125, 0)', 'rgb(197, 151, 0)', 'rgb(49, 86, 220)', 'rgb(148, 69, 117)', 'rgb(236, 255, 220)', 'rgb(210, 76, 178)', 'rgb(151, 112, 60)', 'rgb(76, 37, 127)', 'rgb(158, 3, 102)', 'rgb(136, 255, 236)', 'rgb(181, 100, 129)', 'rgb(57, 109, 43)', 'rgb(86, 115, 95)', 'rgb(152, 131, 118)', 'rgb(155, 177, 149)', 'rgb(169, 121, 92)', 'rgb(228, 197, 211)', 'rgb(159, 79, 103)', 'rgb(30, 43, 57)', 'rgb(102, 67, 39)', 'rgb(175, 206, 120)', 'rgb(50, 46, 223)', 'rgb(134, 180, 135)', 'rgb(194, 48, 0)', 'rgb(171, 232, 107)', 'rgb(150, 101, 109)', 'rgb(37, 14, 53)', 'rgb(166, 0, 25)', 'rgb(0, 128, 207)', 'rgb(202, 239, 255)', 'rgb(50, 63, 97)', 'rgb(164, 73, 220)', 'rgb(106, 157, 59)', 'rgb(255, 90, 228)', 'rgb(99, 106, 1)', 'rgb(209, 108, 218)', 'rgb(115, 96, 96)', 'rgb(255, 186, 173)', 'rgb(211, 105, 180)', 'rgb(255, 222, 214)', 'rgb(108, 109, 116)', 'rgb(146, 125, 94)', 'rgb(132, 93, 112)', 'rgb(91, 98, 193)', 'rgb(47, 74, 54)', 'rgb(228, 95, 53)', 'rgb(255, 59, 83)', 'rgb(172, 132, 221)', 'rgb(118, 41, 136)', 'rgb(112, 236, 152)', 'rgb(64, 133, 67)', 'rgb(44, 53, 51)', 'rgb(46, 24, 45)', 'rgb(50, 57, 37)', 'rgb(25, 24, 27)', 'rgb(47, 46, 44)', 'rgb(2, 60, 50)', 'rgb(155, 158, 226)', 'rgb(88, 175, 173)', 'rgb(92, 66, 77)', 'rgb(122, 197, 166)', 'rgb(104, 93, 117)', 'rgb(185, 188, 189)', 'rgb(131, 67, 87)', 'rgb(26, 123, 66)', 'rgb(46, 87, 170)', 'rgb(229, 81, 153)', 'rgb(49, 110, 71)', 'rgb(205, 0, 197)', 'rgb(106, 0, 77)', 'rgb(127, 187, 236)', 'rgb(243, 86, 145)', 'rgb(215, 197, 74)', 'rgb(98, 172, 183)', 'rgb(203, 161, 188)', 'rgb(162, 138, 154)', 'rgb(108, 63, 59)', 'rgb(255, 228, 125)', 'rgb(220, 186, 227)', 'rgb(95, 129, 109)', 'rgb(58, 64, 74)', 'rgb(125, 191, 50)', 'rgb(230, 236, 220)', 'rgb(133, 44, 25)', 'rgb(40, 83, 102)', 'rgb(184, 203, 156)', 'rgb(14, 13, 0)', 'rgb(75, 93, 86)', 'rgb(107, 84, 63)', 'rgb(226, 113, 114)', 'rgb(5, 104, 236)', 'rgb(46, 181, 0)', 'rgb(210, 22, 86)', 'rgb(239, 175, 255)', 'rgb(104, 32, 33)', 'rgb(45, 32, 17)', 'rgb(218, 76, 255)', 'rgb(112, 150, 142)', 'rgb(255, 123, 125)', 'rgb(74, 25, 48)', 'rgb(232, 194, 130)', 'rgb(231, 219, 188)', 'rgb(166, 132, 134)', 'rgb(31, 38, 60)', 'rgb(54, 87, 78)', 'rgb(82, 206, 121)', 'rgb(173, 170, 169)', 'rgb(138, 159, 69)', 'rgb(101, 66, 210)', 'rgb(0, 251, 140)', 'rgb(93, 105, 123)', 'rgb(204, 210, 127)', 'rgb(148, 165, 161)', 'rgb(121, 2, 41)', 'rgb(227, 131, 230)', 'rgb(126, 164, 193)', 'rgb(78, 68, 82)', 'rgb(75, 44, 0)', 'rgb(98, 11, 112)', 'rgb(49, 76, 30)', 'rgb(135, 74, 166)', 'rgb(227, 0, 145)', 'rgb(102, 70, 10)', 'rgb(235, 154, 139)', 'rgb(234, 195, 163)', 'rgb(152, 234, 179)', 'rgb(171, 145, 128)', 'rgb(184, 85, 47)', 'rgb(26, 43, 47)', 'rgb(148, 221, 197)', 'rgb(157, 140, 118)', 'rgb(156, 131, 51)', 'rgb(148, 169, 201)', 'rgb(57, 41, 53)', 'rgb(140, 103, 94)', 'rgb(204, 233, 58)', 'rgb(145, 113, 0)', 'rgb(1, 64, 11)', 'rgb(68, 152, 150)', 'rgb(28, 163, 112)', 'rgb(224, 141, 167)', 'rgb(139, 74, 78)', 'rgb(102, 119, 118)', 'rgb(70, 146, 173)', 'rgb(103, 189, 168)', 'rgb(105, 37, 92)', 'rgb(211, 191, 255)', 'rgb(74, 81, 50)', 'rgb(126, 146, 133)', 'rgb(119, 115, 60)', 'rgb(231, 160, 204)', 'rgb(81, 162, 136)', 'rgb(44, 101, 106)', 'rgb(77, 92, 94)', 'rgb(201, 64, 58)', 'rgb(221, 215, 243)', 'rgb(0, 88, 68)', 'rgb(180, 162, 0)', 'rgb(72, 143, 105)', 'rgb(133, 129, 130)', 'rgb(212, 233, 185)', 'rgb(61, 115, 151)', 'rgb(202, 232, 206)', 'rgb(214, 0, 52)', 'rgb(170, 103, 70)', 'rgb(158, 85, 133)', 'rgb(186, 98, 0)'
];

var category = null, category_colors = null, name2cat = {};
var MODEL_NAMES = []

Date.prototype.format = function (format) {
  var o = {
    'M+': this.getMonth() + 1, //month
    'd+': this.getDate(), //day
    'h+': this.getHours(), //hour
    'm+': this.getMinutes(), //minute
    's+': this.getSeconds(), //second
    'q+': Math.floor((this.getMonth() + 3) / 3), //quarter
    S: this.getMilliseconds() //millisecond
  };
  if (/(y+)/.test(format)) format = format.replace(RegExp.$1, (this.getFullYear() + '').substr(4 - RegExp.$1.length));
  for (var k in o) {
    if (new RegExp('(' + k + ')').test(format)) format = format.replace(RegExp.$1, RegExp.$1.length == 1 ? o[k] : ('00' + o[k]).substr(('' + o[k]).length));
  } return format;
};

var MAX_LINE_WIDTH = 20;

function setColor(color) {
  graph.setCurrentColor(color);
  $('#color-drop-menu .color-block').css('background-color', color);
  $('#color-drop-menu .color-block').css('border', color == 'white' ? 'solid 1px rgba(0, 0, 0, 0.2)' : 'none');
}

function setCategory(color) {
  labelgraph.setCurrentColor(color);
  $('#category-drop-menu .color-block').css('background-color', color);
  $('#category-drop-menu .color-block').css('border', color == 'white' ? 'solid 1px rgba(0, 0, 0, 0.2)' : 'none');
}

function setLineWidth(width) {
  graph.setLineWidth(width * 2);
  labelgraph.setLineWidth(width * 2);
  $('#width-label').text(width);
}

function setModel(model) {
  if (loading) return;
  currentModel = model;
  var model_name = MODEL_NAMES[model];
  var sizes = config.models[model_name]["image_size"];
  imwidth = sizes[0];
  imheight = sizes[1];
  category = name2cat[model_name][0];
  category_colors = name2cat[model_name][1];
  $('#model-label').text(MODEL_NAMES[model]);
  onStart();
}

function setImage(data) {
  setLoading(false);
  if (!data || !data.ok) return;
  if (!image) {
    $('#stroke').removeClass('disabled');
    $('#clear-image').prop('hidden', false);
    $('#clear-label').prop('hidden', false);
    $('#option-buttons').prop('hidden', false);
    $('#option-buttons').prop('hidden', false);
    $('#extra-args').prop('hidden', false);
  }
  image = data.img;
  label = data.label;

  //var x = document.getElementById('image-container');
  //x.clientHeight = imheight;
  //x.clientWidth = imwidth;
  var x = document.getElementById('image');
  x.width = imwidth;
  x.height = imheight;
  var x = document.getElementById('label');
  x.width = imwidth;
  x.height = imheight;
  var image_stroke = graph.getCurrentColor();
  var label_stroke = labelgraph.getCurrentColor();
  graph.setSize(imheight, imwidth);
  labelgraph.setSize(imheight, imwidth);
  graph.setCurrentColor(image_stroke);
  labelgraph.setCurrentColor(label_stroke);

  $('#image').attr('src', image);
  $('#canvas').css('background-image', 'url(' + image + ')');
  $('#label').attr('src', label);
  $('#label-canvas').css('background-image', 'url(' + label + ')');
  graph.setHasImage(true);
  labelgraph.setHasImage(true);
  spinner.spin();
}

function setLoading(isLoading) {
  loading = isLoading;
  graph.setHasImage(false);
  labelgraph.setHasImage(false);
  $('#start-new').prop('disabled', loading);
  $('#sample-noise').prop('disabled', loading);
  $('#submit').prop('disabled', loading);
  $('#spin').prop('hidden', !loading);
  if (loading) {
    $('.image').css('opacity', 0.7);
    spinner.spin(document.getElementById('spin'));
  } else {
    $('.image').css('opacity', 1);
    spinner.stop();
  }
}

function onSubmit() {
  if (graph && !loading) {
    setLoading(true);
    var formData = {
      model: MODEL_NAMES[currentModel],
      image_stroke: graph.getImageData(),
      label_stroke: labelgraph.getImageData()
    };
    $.post('edit/stroke', formData, setImage, 'json');
  }
}

function onStartNew() {
  if (loading) return;
  setLoading(true);
  graph.clear();
  labelgraph.clear();
  $.post('edit/new', { model: MODEL_NAMES[currentModel] }, setImage, 'json');
}

function onSampleNoise() {
  if (graph && !loading) {
    setLoading(true);
    graph.clear();
    labelgraph.clear();
    $.post('edit/sample-noise', { model: MODEL_NAMES[currentModel] }, setImage, 'json');
  }
}

function onStart() {
  clearMenu();
  initMenu();
  onStartNew();
  $('#start').prop('hidden', true);
}

function onUpload() {
  // filename = $('#choose').val();
  // if (!filename) {
  //   alert('未选择文件');
  //   return false;
  // }
  // $.get(filename, function(data) {
  //   console.log(data);
  // });
  // return false;
}

function onChooseFile(e) {
  // filename = $('#choose').val();
  // console.log(filename);
  // $('.custom-file-control').content(filename);
  // console.log($('.custom-file-control').after());
}

function clearMenu() {
  // $('#color-menu').empty();
  $('#category-menu').empty();
}

function initMenu() {
  // TODO: this need to be fixed to painter color
  /*
  painter_colors.forEach(function (color) {
    $('#color-menu').append(
      '\n<li role="presentation">\n  <div onclick="setColor(\'' +
      color +
      '\')"\n  >\n    <div class="color-block" style="background-color:' +
      color + ';border:' +
      (color == 'white' ? 'solid 1px rgba(0, 0, 0, 0.2)' : 'none') +
      '"/>\n  </div>\n</li>');
  });
  */

  category_colors.forEach(function (color) {
    $('#category-menu').append(
      '\n<li role="presentation">\n  <div onclick="setCategory(\'' +
      color +
      '\')"\n  >\n    <div class="color-block" style="background-color:' +
      color + ';border:' +
      (color == 'white' ? 'solid 1px rgba(0, 0, 0, 0.2)' : 'none') +
      '"/>\n  </div>\n</li>');
  });

  /*
  category_colors.forEach(function (color, idx) {
    $('#category-menu').append(
      '\n<li role="presentation">\n' +
      ' <div style="float:left;width:100%" onclick="setCategory(\'' + 
      color + '\')">\n' + 
      '   <div class="color-block" style="float:left;background-color:' + 
      color + ';border:' + 
      (color == 'white' ? 'solid 1px rgba(0, 0, 0, 0.2)' : 'none') + '"/>\n' +
      '   <div class="semantic-block" >' + 
      category[idx] + '</div>\n</div>\n</li>');
  });
  */
}

function onClearImage() {
  graph.clear();
  labelgraph.clear();
}

function init() {
  initMenu();
  MODEL_NAMES.forEach(function (model, idx) {
    $('#model-menu').append(
      '<li role="presentation">\n' +
      '  <div class="dropdown-item model-item" onclick="setModel(' + idx + ')">' +
      model +
      '  </div>\n' +
      '</li>\n'
    );
  });

  var slider = document.getElementById('slider');
  noUiSlider.create(slider, {
    start: MAX_LINE_WIDTH / 2,
    step: 1,
    range: {
      'min': 1,
      'max': MAX_LINE_WIDTH
    },
    behaviour: 'drag-tap',
    connect: [true, false],
    orientation: 'vertical',
    direction: 'rtl',
  });
  slider.noUiSlider.on('update', function () {
    setLineWidth(parseInt(slider.noUiSlider.get()));
  });

  setColor('black');
  setCategory(category_colors[0]);
  setLineWidth(MAX_LINE_WIDTH / 2);

  $('#download-sketch').click(function () {
    download(
      graph.getImageData(),
      'sketch_' + new Date().format('yyyyMMdd_hhmmss') + '.png');
  });
  $('#download-image').click(function () {
    download(
      image,
      'image_' + new Date().format('yyyyMMdd_hhmmss') + '.png');
  });
  $('#download-doodle').click(function () {
    download(
      labelgraph.getImageData(),
      'doodle_' + new Date().format('yyyyMMdd_hhmmss') + '.png');
  });
  $('#download-label').click(function () {
    download(
      label,
      'label_' + new Date().format('yyyyMMdd_hhmmss') + '.png');
  });
  $('#clear-image').click(onClearImage);
  $('#clear-label').click(labelgraph.clear);
  $('#submit').click(onSubmit);
  $('#stroke').click(function () {
    var stroke = $('#stroke').hasClass('active');
    if (stroke) {
      $('#image').prop('hidden', false);
      $('#label').prop('hidden', false);
      $('#canvas').prop('hidden', true);
      $('#label-canvas').prop('hidden', true);
      $('#stroke').removeClass('active');
      $('#stroke .btn-text').text('Show stroke');
    } else {
      $('#image').prop('hidden', true);
      $('#label').prop('hidden', true);
      $('#canvas').prop('hidden', false);
      $('#label-canvas').prop('hidden', false);
      $('#stroke').addClass('active');
      $('#stroke .btn-text').text('Hide stroke');
    }
  });
  $('#start-new').click(onStartNew);
  $('#sample-noise').click(onSampleNoise);
  $('#start').click(onStart);
}

function download(data, filename) {
  var link = document.createElement('a');
  link.href = data;
  link.download = filename;
  link.click();
}

$(document).ready(function () {
  // get config
  readTextFile("static/config.json",
    function (text) {
      config = JSON.parse(text);
      MODEL_NAMES = Object.keys(config.models);
      for (const model_name of MODEL_NAMES) {
        var n_class = config.models[model_name].n_class;
        var cnames = [], ccolors = [];
        for (var i = 1; i <= n_class; i++) {
          ccolors.push(HIGH_CONSTRAST[i - 1]);
          cnames.push("Label " + i);
        }
        name2cat[model_name] = [cnames, ccolors];
      }
      var key = MODEL_NAMES[0];
      var sizes = config.models[key]["image_size"];
      imwidth = sizes[0];
      imheight = sizes[1];
      category = name2cat[key][0];
      category_colors = name2cat[key][1];
      document.getElementById('model-label').textContent = key;
      graph = new Graph(document, 'canvas');
      labelgraph = new Graph(document, 'label-canvas');
      /*
      var x = document.getElementById('image-container');
      var h = x.clientHeight;
      var w = x.clientWidth;
      console.log("client:", h, w);
      x = document.getElementById('image');
      console.log("image:", x.height, x.width);
      h = x.height / x.width * w;
      x.width = h;
      x.height = w;
      x = document.getElementById('label');
      x.width = h;
      x.height = w;
      graph.setSize(h, w);
      labelgraph.setSize(h, w);
      */
      init();
    });
});
