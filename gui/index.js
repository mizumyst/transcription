var lyricsData;
var player, lyricsContainer;
var prevIndices = "";

var source_json;

const epsilon = 1e-6;

function throttle(callback, delay) {
    let shouldWait = false;

    return (...args) => {
        if (shouldWait) return;

        callback(...args);
        shouldWait = true;
        setTimeout(() => {
            shouldWait = false;
        }, delay);
    };
}

// t: type of annotation (text, gloss, note)
function makeEditableAnnotation(e) {
    var editableAnnotation;

	$( e.target ).replaceWith(function(){
		editableAnnotation = $("<textarea type='text' autocomplete='off' />").attr({
			id: $(this).attr("id"),
		});
		editableAnnotation.val($(this).html());
        editableAnnotation.addClass(e.target.className);

		editableAnnotation.on("blur", (z) => {
			const parts = z.target.value.split("//").map(item => item.trim());
			lyricsData[e.target.id][z.target.classList[0]] = parts[0];
			if (parts.length == 1) {
				updateAnnotations(edited=true);
				return;
			}
			
			const currentSpeaker = lyricsData[e.target.id].speaker;
			const currentEnd = lyricsData[e.target.id].end;
			parts.splice(1).forEach((part, index) => {
                newAnnotation(
                    part, e.target.id + index,
                    player.currentTime - epsilon,
                    currentEnd,
                    speaker=currentSpeaker
                );
			})
			lyricsData[e.target.id].end = player.currentTime + epsilon;
			updateAnnotations(edited=true);
		});
		return editableAnnotation;
	});

    editableAnnotation.focus();
}


// TODO: forced lyricClass is hacky
function newAnnotation(part, index, start, end, speaker=0, lyricClass="text") {
    const newLyric = {
        speaker: speaker,
        start: start,
        end: end
    };
    newLyric[lyricClass] = part;
    lyricsData.splice(index, 0, newLyric);
}


function createNewAnnotationHere() {
    var lastIndex = 0;
    lyricsData.forEach((item, index) => {
		if (item.start < player.currentTime && player.currentTime < item.end)
			lastIndex = index;	
	});
    newAnnotation("", lastIndex, player.currentTime - 5, player.currentTime + 5)
    updateAnnotations(edited=true);
}


function makeClickableToggle(e) {
    var newValue = lyricsData[e.target.id][e.target.classList[0]];
    newValue = newValue == '[UNK]' ? 0 : (newValue + 1) % 3;
    e.target.innerHTML = newValue;
    lyricsData[e.target.id][e.target.classList[0]] = newValue;
}


function addAnnotationsLine(item, index) {
    var row = $("<tr />");

	var speaker = $(" <td class='speaker' /> ").attr({
		id: index
	});
	speaker.on("click", makeClickableToggle);
	speaker.append(document.createTextNode(item.speaker));

	var text = $(" <td class='text w-100' /> ").attr({
		id: index
	});
	text.on("click", makeEditableAnnotation);
	text.append(document.createTextNode(item.text));

    // var gloss = $(" <td class='lyricsText' /> ").attr({
	// 	id: index
	// });
	// gloss.on("click", makeEditableAnnotation);
	// gloss.append(document.createTextNode(item.gloss));

    // var note = $(" <td class='lyricsText' /> ").attr({
	// 	id: index
	// });
	// note.on("click", makeEditableAnnotation);
	// note.append(document.createTextNode(item.note));

    row.append(speaker);
    row.append(text);
    // row.append(gloss);
    // row.append(note);

	lyricsContainer.append(row);
}

function download() {
    const link = document.createElement("a");
    const file = new Blob([JSON.stringify(lyricsData)], { type: 'text/plain' });
    link.href = URL.createObjectURL(file);
    link.download = source_json;
    link.click();
    URL.revokeObjectURL(link.href);
}

function round() {
	lyricsData.forEach((item) => {
		item.start = Math.round(item.start * 1000) / 1000;
		item.end = Math.round(item.end * 1000) / 1000;
	});
}

function updateAnnotations(edited=false) {
	var indicesRaw = [];
	lyricsData.forEach((item, index) => {
		if (item.start < player.currentTime + 1 && player.currentTime - 1 < item.end)
			indicesRaw.push(index);	
	});
	currIndices = indicesRaw.toString();
	if (!edited && currIndices == prevIndices)
		return;
	
	prevIndices = currIndices;
	
	lyricsContainer.empty();
	indicesRaw.forEach((index) => {
		addAnnotationsLine(lyricsData[index], index);
	})
}


function setSource() {
    source = $("#filename").val();

    player.src = source;
    player.load();

    source_json = source + ".json";

	fetch(source_json)
		.then((res) => res.json())
		.then((result) => {
			lyricsData = result;
			player.addEventListener('timeupdate', throttle(updateAnnotations, 300));
		})
		.catch((e) => console.error(e));
    
    source_json = source_json.split("/").slice(-1);
}

$( window ).on('keydown', function(event) {
    if (event.ctrlKey || event.metaKey) {
        switch (event.key) {
            case 's':
                event.preventDefault();
				round();
                download();
                break;
			case ' ':
				event.preventDefault();
				player.paused ? player.play() : player.pause();
				break;
		}
    }
});

$(function(event) {
	player = document.querySelector('.player');
	lyricsContainer = $('.lyricsContainer');
    
    $("#filename").val("../data/mini1.wav");
    setSource()
});

